# Setup Atlas

## DGX3

### Setup
```bash
INSTALLATION_DIR="toolformer"
cd $INSTALLATION_DIR
gtt clone git@github.com:lamarr-nlp/atlas.git
python3 -m virtualenv --prompt Atlas --system-site-packages "pyenv_atlas"
source "pyenv_atlas/bin/activate"
pip install --upgrade pip
pip install -r atlas/requirements.txt
```


## Download Natural Questions (NQ) Index and Passages, Data and Atlas-Base Model
> Already done for DGX2!
> There are 127 embedding and passages file pairs to download the base index. Roughly 200 GB disk space is needed!

```bash
cd atlas
mkdir -p "downloads/experiments/"
python preprocessing/download_model.py --model models/atlas_nq/large --output_directory "/raid/s3/opengptx/atlas_resources/downloads"
python preprocessing/download_index.py --index indices/atlas_nq/wiki/large --output_directory "/raid/s3/opengptx/atlas_resources/downloads"
python preprocessing/prepare_qa.py --output_directory "downloads"
```

## Test Setup in VSCode `launch.json` or bash

```bash
-m torch.distributed.run --standalone --nnodes 1 --nproc_per_node 1 atlas/inference.py --name dev-eval-large-nq --generation_max_length 32 --target_maxlength 32 --gold_score_mode ppmean --precision fp32 --reader_model_type google/t5-large-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path /raid/s3/opengptx/atlas_resources/downloads/models/atlas_nq/large --eval_data /raid/s3/opengptx/atlas_resources/downloads/nq_data/debug.jsonl --load_index_path /raid/s3/opengptx/atlas_resources/downloads/indices/atlas_nq/wiki/large --per_gpu_batch_size 1 --n_context 40 --retriever_n_context 40 --checkpoint_dir /raid/s3/opengptx/atlas_resources/downloads/experiments/ --load_index_path /raid/s3/opengptx/atlas_resources/downloads/indices/atlas_nq/wiki/large --index_mode flat --save_index_n_shards 4 --task qa --write_results 
```

```json
{
            "name": "Torchrun Atlas Inference",
            "type": "python",
            "request": "launch",
            "module": "torch.distributed.run",
            "env": {
                "CUDA_VISIBLE_DEVICES": "2"
            },
            "args": [
                "--standalone",
                "--nnodes",
                "1",
                "--nproc_per_node",
                "1",
                "atlas/inference.py",
                "--name",
                "dev-eval-large-nq",
                "--generation_max_length",
                "32",
                "--target_maxlength",
                "32",
                "--gold_score_mode",
                "ppmean",
                "--precision",
                "fp32",
                "--reader_model_type",
                "google/t5-large-lm-adapt",
                "--text_maxlength",
                "512",
                "--target_maxlength",
                "16",
                "--model_path",
                "/raid/s3/opengptx/atlas_resources/downloads/models/atlas_nq/large",
                "--eval_data",
                "/raid/s3/opengptx/atlas_resources/downloads/nq_data/debug.jsonl",
                "--load_index_path",
                "/raid/s3/opengptx/atlas_resources/downloads/indices/atlas_nq/wiki/large",
                "--per_gpu_batch_size",
                "1",
                "--n_context",
                "40",
                "--retriever_n_context",
                "40",
                "--checkpoint_dir",
                "/raid/s3/opengptx/atlas_resources/downloads/experiments/",
                "--load_index_path",
                "/raid/s3/opengptx/atlas_resources/downloads/indices/atlas_nq/wiki/large",
                "--index_mode",
                "flat",
                "--save_index_n_shards",
                "4",
                "--task",
                "qa",
                "--write_results"
            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "envFile": "${workspaceFolder}/.env",
            "cwd": "${workspaceFolder}"
        }
```