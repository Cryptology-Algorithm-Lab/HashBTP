### General Security Analysis for Face Template Protection Methods from Cryptographic Hash Functions

- Seunghun Paik, Minsu Kim, Sunpill Kim, and Jae Hong Seo

- Department of Mathematics & Research Institute for Natural Sciences, Hanyang University

- Submitted to IEEE Transactions on Information Forensics and Security.

#### Introduction

This codebase includes the scripts to reproduce our experimental results and the proposed security estimator.

#### Prerequisites

All the required libraries are in `dependencies.yml`, such as PyTorch, SciPy, and Galois

#### Structure of the Codebase

The overall structure of our codebase is as follows: (To be written.)

```
HashBTP
├──backbones
    ├── __init__.py
    ├── custom_layers.py              # Implementation of SLL; Forked from https://github.com/araujoalexandre/lipschitz-sll-networks.
    ├── iresnet.py                    # Forked from https://github.com/deepinsight/insightface    
    ├── iresnet_AT.py                 # Due to compatibility issue; Forked from https://github.com/ShawnXYang/Face-Robustness-Benchmark
    ├── toyexamples.py                # Networks for toy examples on our analysis.
    └── sllnet.py   
├── config.py                       # Configs for loading pre-trained FR models.
├── BTPs.py                         # Implemnetation of face BTPs, including IronMask (CVPR'21) and Mohan et al. (CVPRW'19)
├── estimator.py                    # Implementation of the proposed security estimator.
├── benchmark.py                    # Codes for measuring TAR/FAR.
├── feat_tools.py                   # Utility tools for extracting face features from the benchmark datasets.
├── 1. Upper TAR.ipynb              # Reproducing our results for the upper bound of TAR from CtH-based BTPs
└── 2. Estimator.ipynb              # Reproducing our results for the proposed security estimator
```

#### 1. Reproducing Upper Bounds on TAR

1. You need to download pre-trained parameters for FR models. We used the following open-sourced FR libraries whose parameters are publicly available.

- InsightFace: https://github.com/deepinsight/insightface/tree/master/model_zoo

- OpenSphere: https://github.com/ydwen/opensphere 

- CVLFace: https://github.com/mk-minchul/CVLface

In addition, we also used the official implementations of other recent FR models with pre-trained parameters.

- UniTSFace: https://github.com/CVI-SZU/UniTSFace 

- ElasticFace: https://github.com/fdbtrs/ElasticFace

- Face-Transformer: https://github.com/zhongyy/Face-Transformer

The codes for loading these backbones are provided inside `backbone`. You can check details at `backbone/__init__.py`. For the detailed settings about each FR model, e.g., architecture, loss function, or train dataset, please refer to `Table 1` of our paper. We also provide the Google Drive link that contains all the parameters used in our paper: https://drive.google.com/file/d/1pYLZas2NgGMglW71QSSLOpIaO_inR-WA/view?usp=sharing

2. You also need to download face benchmark datasets (LFW, CFP-FP, and AgeDB). You can obtain them by downloading them from their official websites or by downloading one of the training datasets provided by InsightFace:

- https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

3. Now you can reproduce our experimental results by following the instructions in `1. Upper TAR.ipynb`.


#### 2. Reproducing Results for Security Estimator

1. The target BTPs are implemented in `BTPs.py`, and the proposed security estimator is provided in `estimator.py`.

2. You can run our estimator on these BTPs by following the instructions in `2. Estimator.ipynb`.


#### Enjoy!
