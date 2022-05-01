# Aspect-Based-Document-Similarity
# Aspect-based Document Similarity for Research Papers

Implementation, trained models and result data for the paper **Aspect-based Document Similarity for Research Papers** [(PDF on Arxiv)](https://arxiv.org/abs/2010.06395). 
The supplemental material is available for download under [GitHub Releases](https://github.com/malteos/aspect-document-similarity/releases) or [Zenodo](http://doi.org/10.5281/zenodo.4087898).

- Datasets are compatible with 🤗 [Huggingface NLP library](https://github.com/huggingface/nlp). 


<img src="https://raw.githubusercontent.com/malteos/aspect-document-similarity/master/docrel.png">


## Requirements

- Python 3.7
- CUDA GPU (for Transformers)
- For A100 GPU (Cuda 11.2) Pytorch is not built. So we built it from source. For reference please follow the  [link](https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1)
## Installation

Create a new virtual environment for Python 3.7 with Conda:
 
 ```bash
conda create -n vm python=3.7
conda activate vm
```

Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

Inside this environment build Pytorch from the link above.

## Dataset for BERT Model
Same dataset as used in the paper is used for getting the results which contains only title and abstract in the input for every paper.

## Creating Dataset for Longformer
The given dataset does not include "body_text" so it needs to be merged from another dataset. 
Here is the link of the latest dataset which contains body_text but the format of the data is different from the given data. 

The given script "merge_dataset.py" can be run after downloading the latest dataset and extracting it from the file. 

The python file "merge_dataset.py" takes the given data from docs.jsonl and it matches the cord19-id with the json files given in the latest dataset in the folder "document_parses/pdf_json". All the paper id were present in the latest dataset and the "text" keys of "body_text" is appended and merged with the given dataset. 

This script will give us the merged dataset which contains the full research paper. 

## Experiments

To reproduce our experiments, follow these steps:

### Prepare

```bash
export DIR=./output

# CORD-19
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-03-13.tar.gz

# Get additional data (collected from Semantic Scholar API)
wget https://github.com/malteos/aspect-document-similarity/releases/download/1.0/cord19_s2.tar
```

### Build datasets

```bash
# CORD-19
python -m cord19.dataset save_dataset <input_dir> <output_dir>

```

### Use dataset

The datasets are built on the Huggingface NLP library (soon available on the official repository):

```python
from nlp import load_dataset

# Training data for first CV split
train_dataset = load_dataset(
    './datasets/cord19_docrel/cord19_docrel.py',
    name='relations',
    split='fold_1_train'
)                   
```

### Train models

All models are trained with the `trainer_cli.py` script:

```bash
python trainer_cli.py --cv_fold $CV_FOLD \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME \
    --doc_id_col $DOC_ID_COL \
    --doc_a_col $DOC_A_COL \
    --doc_b_col $DOC_B_COL \
    --nlp_dataset $NLP_DATASET \
    --gradient_accumulation_steps $GRAD_ACC_STEP \
    --nlp_cache_dir $NLP_CACHE_DIR \
    --cache_dir $CACHE_DIR \
    --num_train_epochs $EPOCHS \
    --seed $SEED \
    --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LR \
    --logging_steps 100 \
    --save_steps 5000 \
    --save_total_limit 3 \
    --do_train \
    --save_predictions
```
We used $MODEL_NAME = allenai/longformer-base-4096
The exact parameters are available in `sbin/cord19/longformer.sh`. 




