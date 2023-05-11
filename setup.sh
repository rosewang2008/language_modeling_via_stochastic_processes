#!/bin/bash

conda create -n language_modeling_via_stochastic_processes python=3.8.8
conda activate language_modeling_via_stochastic_processes
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 torchmetrics==0.2.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c intel mkl_random==1.1.1
pip install -r requirements.txt