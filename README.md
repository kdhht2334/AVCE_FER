# AVCE_FER
> **Emotion-aware Multi-view Contrastive Learning for Facial Emotion Recognition (ECCV 2022)**<br>

<a href="https://releases.ubuntu.com/16.04/"><img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-16.04-green"></a>
<a href="https://www.python.org/downloads/release/python-370/"><img alt="PyThon" src="https://img.shields.io/badge/Python-v3.8-blue"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

[Daeha Kim](https://scholar.google.co.kr/citations?user=PVt7f0YAAAAJ&hl=ko), [Byung Cheol Song](https://scholar.google.co.kr/citations?user=yo-cOtMAAAAJ&hl=ko)

CVIP Lab, Inha University


## Real-time demo with pre-trained weights
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


## News

[22.07.10]: Add source code and demo.

[22.07.07]: OPEN official pytorch version of AVCE_FER.


## Datasets

1. Download three public benchmarks for training and evaluation (I cannot upload datasets due to the copyright issue).

  - [AffectNet](http://mohammadmahoor.com/affectnet/)
  - [Aff-wild](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) 
  - [Aff-wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
  - [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
 
 (For more details visit [website](https://ibug.doc.ic.ac.uk/))

2. Follow preprocessing rules for each dataset by referring pytorch official [custom dataset tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).


## Pretrained weights

* Check `pretrained_weights` folder.

  - Weights are trained on AFEW-VA dataset.
  
  - Weights for __demo__ are trained on multiple VA database (please refer [here](https://github.com/kdhht2334/AVCE_FER/tree/main/AVCE_demo))


## Run

1. Go to `/src`.

2. Train AVCE.

3. (Or) Execute `run.sh`

```python
CUDA_VISIBLE_DEVICES=0 python main.py --freq 250 --model alexnet --online_tracker 1 --data_path <data_path> --save_path <save_path>
```

| Arguments | Description
| :-------- | :--------
| freq | Parameter saving frequency.
| model | CNN model for backbone. Choose from 'alexnet', and 'resnet18'.
| online_tracker | Wandb on/off.
| data_path | Path to load facial dataset.
| save_path | Path to save weights.



## Real-time demo

1. Go to `/AVCE_demo`.

2. Run `main.py`.

  - Facial detection and AV FER functionalities are equipped.
  - Before that, you have to train and save `Encoder.t7` and `FC_layer.t7`.


## Citation

	@inproceedings{kim2022emotion,
    	title={Emotion-aware Multi-view Contrastive Learning for Facial Emotion Recognition},
    	author={Kim, Daeha and Song, Byung Cheol},
    	booktitle={European Conference on Computer Vision},
    	pages={178--195},
    	year={2022},
    	organization={Springer}
  }


### Contact
If you have any questions, feel free to contact me at `kdhht5022@gmail.com`.