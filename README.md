# Improving Fault Localization using Test Cases with High Fault Diagnosis Capability
This repository contains a replication package for a research paper submitted to ASE 2023.

## Requirements:
+ Python 3.9.1
  + ``` pip install -r requirements.txt ```
  + PyTorch with CUDA
+ Linux/amd64 architecture
+ Docker 

## Package structure
+ **human-written-tests** directory contains code and data that can help reproduce results for RQ1
+ **automatically-generated-tests** directory contains code that can help reproduce results for RQ2-3

## Acknowledgement
**We benefit a lot from the following projects when building our technique**
+ [FDG](https://github.com/agb94/FDG-artifact)
+ [DDU](https://github.com/aperez/evosuite)
+ [EVOSUITE](http://www.evosuite.org)