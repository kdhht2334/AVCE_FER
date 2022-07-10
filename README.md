# AVCE_FER
This repository provides the official PyTorch implementation of the following paper:

> **Emotion-aware Multi-view Contrastive Learning for Facial Emotion Recognition (ECCV 2022)**<br>

- Real-time demo with pre-trained weights
<p align="center">
<img src="https://github.com/kdhht2334/AVCE_FER/blob/main/AVCE_demo/AVCE_demo_vid.gif" height="320"/>
</p>

## Requirements

- Python (>=3.7)
- PyTorch (>=1.7.1)
- pretrainedmodels (>=0.7.4)
- cvxpy (>=1.1.15)
- [Wandb](https://wandb.ai/)
- [Fabulous](https://github.com/jart/fabulous) (terminal color toolkit)

To install all dependencies, do this.

```
pip install -r requirements.txt
```

-------

## News

__[22.07.10]: Add source code and demo.__

__[22.07.07] OPEN official pytorch version of AVCE_FER.__

-------

## Datasets

1. Download three public benchmarks for training and evaluation (I cannot upload datasets due to the copyright issue).

  - [AffectNet](http://mohammadmahoor.com/affectnet/)
  - [Aff-wild](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) 
  - [Aff-wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
  - [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
 
 (For more details visit [website](https://ibug.doc.ic.ac.uk/))

2. Follow preprocessing rules for each dataset by referring pytorch official [custom dataset tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

-------

## Run

1. Go to `/src`.

2. Train AVCE.

```python
CUDA_VISIBLE_DEVICES=0 python main.py --freq 250 --model alexnet --online_tracker 1 --data_path <data_path> --save_path <save_path>
```

- Arguments
 - __freq__: Parameter saving frequency
 - __model__: CNN model for backbone.
 - __online_tracker__: Wandb on/off.
 - __data_path__: Path to load facial dataset.
 - __save_path__: Path to save weights.
 
 -------

## Real-time demo

1. Go to `/Real_demo`.

2. Run `main.py`.

  - Facial detection and AV FER functionalities are equipped.
  - Before that, you have to train and save `Encoder.t7` and `FC_layer.t7`.

---
### BibTeX

Updated soon.
-------
