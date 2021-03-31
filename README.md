# vtb-data-fusion

This repository provides code solution for [Data Fusion Contest](https://boosters.pro/championship/data_fusion/overview) task 1  
**Short description:** `Single distilbert`  
**Place: *7/265***  
**Public LB = *0.8683***  

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets
[Boosters](https://boosters.pro/championship/data_fusion/data)
### Data description
Task is to predict the predefined category of the item in a receipt based on its name


## Solution description
- Baseline — [Russian Part of Multilingual Distillbert](https://huggingface.co/Geotrend/bert-base-ru-cased) as is (spoiler - it was *Cased*): Public = `0.7875`
- **+** Pretraining on **masked language modeling** task: Public = `0.8261`
- **+** *Label Smoothing*: Public = `0.8323`
- **+** *Custom Model Arch* (Weighted sum of hidden states + multisample dropout): Public = `0.8354`
- **+** *Lowercase*: Public = `0.8459`
- **+** *Increase number* of training *epochs* to 50: Public = `0.8532`
- **+** *Pseudolabeling* (distilbert-distilbert): Public = `0.8626`
- **+** *Pseudolabeling* (RuBERT-distilbert): Public = `0.8683`


## How to run
- Pretrain [RuBERT](https://huggingface.co/DeepPavlov/rubert-base-cased) and [distilbert](https://huggingface.co/Geotrend/bert-base-ru-cased) on all unique texts using **masked language modeling** task: `train_mlm_base_tokenizer.ipynb`
- Finetune pretrained **RuBERT** on the texts with labels (~40k unique texts): `rubert_base.ipynb`
- Create pseudolabels (~1M unique texts) for all unique texts using finetuned **RuBERT**: `pseudo_label.ipynb`
- Finetune **distilbert** on these pseudolabels: `pseudo_label.ipynb`
- Create submission *.zip* with finetuned **distilbert**: `pseudo_label.ipynb`