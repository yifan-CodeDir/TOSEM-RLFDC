# Automatically Learning a Precise Measurement for Fault Diagnosis Capability of Test Cases
This repository contains a replication package for a research paper submitted to ISSTA 2024.

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
