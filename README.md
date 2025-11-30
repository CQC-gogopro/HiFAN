# HiFAN

## 1. Make the enriroment
Our code is based on  `pytorch==2.0.0+cu118` `cuda==11.8` `timm==1.0.22` `mmcv==2.0.0`. Other environments can refer to requirements.txt
```
conda create -n HiFAN python=3.10
git clone https://github.com/CQC-gogopro/HiFAN.git
cd HiFAN
pip install requirements.txt   # note: may not all used
```

## 2. Get data 
You can download the PASCAL-Context and NYUD-v2 from ATRC's repository:
### PASCAL-Context
```
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
tar xfvz PASCALContext.tar.gz
```
### NYUD-v2
```
wget https://data.vision.ee.ethz.ch/brdavid/atrc/NYUDv2.tar.gz
tar xfvz NYUDv2.tar.gz
```

**Attention**: You need to set the ```db_root``` variable in ```./configs/mypath.py``` to the root path of all your datasets, with the DATASET_ROOT folder formatted as follows:
```
DATASET_ROOT
├── NYUDv2
└── PASCALContext
```


## 3. Evaluate the model
| Method          | Semseg | Parsing | Saliency | Normal | Edge | Depth | Params | Pretrained_Link |
|-----------------|--------|---------|----------|--------|------|--------|--------|------------------|
| **nyu_swinL**   |58.23   |    -    |    -     |19.10   |78.90 |0.4938  |   297M | [link](https://drive.google.com/file/d/1H_SheK-EEoI3jyrHkLkqRQrx7FHeQMB-/view?usp=sharing)        |
| **pascal_swinB**|78.59   |70.54    |84.88     |14.64   |76.80 |   -    |   146M | [link](https://drive.google.com/file/d/1CiwC875R0ARv_fbt7MbsWV2d0jVqXzuz/view?usp=sharing)        |
| **pascal_swinL**|83.00   |73.71    |85.16     |14.64   |78.20 |   -    |   327M | [link](https://drive.google.com/file/d/1qfawypLeAYAgEOIiuVADjEtD_Ohb1QIV/view?usp=sharing)        |


Our trained HiFAN model can be obtained [here](https://drive.google.com/file/d/1mb07S3Ox85fF0_dUwleDiN7fkjqUmdIN/view?usp=drive_link)
```
bash ./script/infer/nyu_swinL
bash ./script/infer/pascal_swinB
bash ./script/infer/pascal_swinL
```

The evaluation of Boundary Detection Task is based on external [codebase](https://github.com/prismformore/Boundary-Detection-Evaluation-Tools) (which is Matlab-based).
## 4. Train the model
Prepare the pretrained Swin-Large checkpoint by running the following command:
```
cd pretrained_ckpts
bash run.sh
cd ../
```
run the following command for training:
```
bash ./scripts/train/nyu_swinL.sh
bash ./scripts/train/pascal_swinB.sh
bash ./scripts/train/pascal_swinL.sh
```
## Cite
If you found this code/work to be useful in your own research, please cite the following:
```
@article{yu2025parameter,
  title={Parameter Aware Mamba Model for Multi-task Dense Prediction},
  author={Yu, Xinzhuo and Zhuge, Yunzhi and Gong, Sitong and Zhang, Lu and Zhang, Pingping and Lu, Huchuan},
  journal={arXiv preprint arXiv:2511.14503},
  year={2025}
}
```
