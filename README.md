# Automatically Learning a Precise Measurement for Fault Diagnosis Capability of Test Cases
This repository contains a replication package for a research paper submitted to TOSEM.

## Requirements:
+ Python 3.9.1
  + ``` pip install -r requirements.txt ```
  + PyTorch with CUDA
+ Linux/amd64 architecture
+ Docker 

## Package structure
+ **human-written-tests** directory contains code, data and model checkpoints that can help reproduce results for RQ1 and RQ4.
+ **automatically-generated-tests** directory contains code and model checkpoints that can help reproduce results for RQ2-3

## Acknowledgement
**We benefit a lot from the following projects when building our technique**
+ [FDG](https://github.com/agb94/FDG-artifact)
+ [DDU](https://github.com/aperez/evosuite)
+ [EVOSUITE](http://www.evosuite.org)

## Citing
```
@article{10.1145/3712189,
author = {Zhao, Yifan and Sun, Zeyu and Wang, Guoqing and Liang, Qingyuan and Zhang, Yakun and Lou, Yiling and Hao, Dan and Zhang, Lu},
title = {Automatically Learning a Precise Measurement for Fault Diagnosis Capability of Test Cases},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1049-331X},
url = {https://doi.org/10.1145/3712189},
doi = {10.1145/3712189},
note = {Just Accepted},
journal = {ACM Trans. Softw. Eng. Methodol.},
month = jan,
}
```
