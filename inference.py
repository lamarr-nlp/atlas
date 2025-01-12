# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import os
import time
from collections import defaultdict
from typing import Dict, List
from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel
import torch
import torch.cuda
import torch.distributed as dist
import uvicorn
from src.atlas import Atlas
from src.index import DistributedIndex

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_batch_size))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator


@torch.no_grad()
def run_retrieval_only(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)
        retrieved_passages, _ = unwrapped_model.retrieve(
            index,
            opt.n_context,
            query,
            query_enc["input_ids"].cuda(),
            query_enc["attention_mask"].cuda(),
            batch_metadata=batch_metadata,
            filtering_fun=task.filter,
        )
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        for k in range(len(retrieved_passages)):
            if opt.write_results:
                gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
                ex = {"query": query[k], "answers": gold, "passages": retrieved_passages[k]}
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics

@torch.no_grad()
def evaluate(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")
        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)
        if not opt.use_file_passages:
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            retrieved_passages, _ = unwrapped_model.retrieve(
                index,
                opt.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )
        else:
            assert "passages" in batch, "cant use use_file_passages mode without passing in passages"
            retrieved_passages = [p[: opt.n_context] for p in batch["passages"]]

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue

        reader_tokens, _ = unwrapped_model.tokenize_passages(query, retrieved_passages)

        if "eval_loss" in task.metrics:
            eval_loss, logits = unwrapped_model.compute_reader_loss_and_logits(reader_tokens, decoder_input_ids, labels)
            metrics["eval_loss"].append(eval_loss)

        generation = unwrapped_model.generate(
            reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
        )

        for k, g in enumerate(generation):
            if opt.decoder_prompt_format is not None:
                query_ids = reader_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                )
                g = g[len(query_ids) + 1 :]
            pred = reader_tokenizer.decode(g, skip_special_tokens=True)
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            sample_metrics = task.evaluation(pred, gold)
            for key, value in sample_metrics.items():
                metrics[key].append(value)

            if opt.write_results:
                ex = {"query": query[k], "answers": gold, "generation": pred}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics

@torch.no_grad()
def retrieve(queries: List[str], model: Atlas, index: DistributedIndex, opt: Namespace, topk: int) -> List[List[Dict[str, str]]]:
    model.eval()
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    query_enc, _, _ = unwrapped_model.tokenize(queries, [""], target_tokens=None)
    query_ids_retriever = query_enc["input_ids"].cuda()
    query_mask_retriever = query_enc["attention_mask"].cuda()
    retrieved_passages, _ = unwrapped_model.retrieve(
        index,
        topk,
        queries,
        query_ids_retriever,
        query_mask_retriever,
        batch_metadata=None,
        filtering_fun=None,
    )
    return retrieved_passages

@torch.no_grad()
def reader_prediction(queries: List[str], passages: List[List[Dict[str, str]]], model: Atlas) -> List[str]:
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)

    reader_tokens, _ = unwrapped_model.tokenize_passages(queries, passages)
    generation = unwrapped_model.generate(reader_tokens, queries, choices=None)

    predictions = []
    reader_tokenizer = unwrapped_model.reader_tokenizer
    for k, g in enumerate(generation):
        if opt.decoder_prompt_format is not None:
            query_ids = reader_tokenizer.encode(
                opt.decoder_prompt_format.format_map({"query": queries[k]}), add_special_tokens=False
            )
            g = g[len(query_ids) + 1 :]
        predictions.append(reader_tokenizer.decode(g, skip_special_tokens=True))
    return predictions

@torch.no_grad()
def inference_with_retrieval(queries: List[str], model: Atlas, index: DistributedIndex, opt: Namespace, topk: int):
    model.eval()
    retrieved_passages = retrieve(queries=queries, model=model, index=index, opt=opt, topk=topk)
    predictions = reader_prediction(queries=queries, passages=retrieved_passages, model=model)
    return predictions, retrieved_passages


class AtlasOutput(BaseModel):
    generations: List[str]
    passages: List[List[Dict[str, str]]]


class AtlasAPI(FastAPI):
    def __init__(self, model: Atlas, index: DistributedIndex, opt: Namespace, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.index = index
        self.opt = opt

        self.add_api_route("/inference_with_retrieval", self.inference_with_retrieval_api, methods=["POST"], response_model=AtlasOutput)

    def inference_with_retrieval_api(self, queries: List[str], topk: int = 100) -> AtlasOutput:
        logger.info("Inference Request")
        predictions, retrieved_passages = inference_with_retrieval(queries=queries, model=self.model, index=self.index, opt=self.opt, topk=topk)
        logger.info(f"Queries: {queries}")
        logger.info(f"Generations: {predictions}")
        logger.info(f"Passages: {retrieved_passages}")
        return AtlasOutput(generations=predictions, passages=retrieved_passages)
        

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    assert not opt.multi_gpu, "FastAPI serving only supports singel GPU inference!"

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    index, passages = load_or_initialize_index(opt)
    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)

    
    dist_utils.barrier()

    
    if not opt.use_file_passages and opt.load_index_path is None:
        logger.info("Build index")
        indexing_start = time.time()
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

        if opt.save_index_path is not None:
            save_embeddings_and_index(index, opt)

    logger.info("Start Inference")
    # queries = ["Who is Rick Bayless?"]
    
    # predictions = inference_with_retrieval(queries, model, index, opt)
    # logger.info(predictions)
    api = AtlasAPI(model, index, opt)
    uvicorn.run(api, host="localhost", port=8666)