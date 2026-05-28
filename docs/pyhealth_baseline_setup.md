# PyHealth Baseline Setup

Date: 2026-05-20

## Location

```text
medFactoraalpha/pyhealth
```

PyHealth was cloned from:

```text
https://github.com/sunlabuiuc/PyHealth.git
```

Current local commit:

```text
acaf7b7
```

## Conda Environment

Environment name:

```bash
conda activate medfactoraalpha-pyhealth
```

Python:

```text
Python 3.12.13
```

Installed package versions:

```text
pyhealth      2.0.1
torch         2.7.1+cu118
torchvision   0.22.1+cu118
transformers  4.53.3
tokenizers    0.21.4
numpy         2.2.6
```

Note: `pyhealth.__version__` currently reports `2.0.0`, while `pip show pyhealth`
and `pyproject.toml` report `2.0.1`.

## Install Notes

The first `pip install -e .` attempt failed because the default Aliyun PyPI mirror
could not resolve a compatible `tokenizers>=0.21,<0.22` wheel for
`transformers~=4.53.2`.

Working sequence:

```bash
conda create -y -n medfactoraalpha-pyhealth python=3.12 pip
conda activate medfactoraalpha-pyhealth
python -m pip install tokenizers==0.21.4 -i https://pypi.org/simple
python -m pip install -e . -i https://mirrors.aliyun.com/pypi/simple --extra-index-url https://pypi.org/simple
```

The initial PyHealth install pulled `torch==2.7.1+cu126`, but this machine has
NVIDIA driver `520.61.05` and CUDA driver/runtime compatibility up to CUDA 11.8.
That made `torch.cuda.is_available()` return `False`.

The torch stack was replaced with CUDA 11.8 wheels:

```bash
conda activate medfactoraalpha-pyhealth
python -m pip uninstall -y torch torchvision nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
```

After moving PyHealth from `medFactoraalpha/baselines/pyhealth` to the project
top level, the editable install was refreshed:

```bash
cd medFactoraalpha/pyhealth
conda activate medfactoraalpha-pyhealth
python -m pip install -e . --no-deps
```

Current editable location:

```text
medFactoraalpha/pyhealth
```

Current GPU check:

```text
torch 2.7.1+cu118
cuda built 11.8
cuda available True
device_count 4
0 NVIDIA A30
1 NVIDIA A30
2 NVIDIA A30
3 NVIDIA A30
```

`python -m pip check` reports no broken requirements.

## Smoke Tests

Passed:

```bash
python -m unittest tests.core.test_sample_dataset tests.core.test_mlp tests.core.test_logistic_regression -v
```

Result:

```text
Ran 17 tests in 65.727s
OK
```

Also ran a toy MLP training loop using PyHealth dataset, dataloader, model, trainer,
checkpointing, and evaluation.

CPU output:

```text
medFactoraalpha/results/pyhealth_smoke/toy_mlp_1epoch
```

Toy test metrics:

```text
accuracy: 0.25
f1: 0.0
loss: 0.7111
```

GPU output:

```text
medFactoraalpha/results/pyhealth_smoke/toy_mlp_gpu_1epoch
```

GPU toy run confirmed:

```text
trainer_device: cuda:0
param_device: cuda:0
accuracy: 0.5
f1: 0.0
loss: 0.7091
```
