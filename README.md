

# Divide-and-Conquer Grouping Distillation


# MDistiller

### Introduction

MDistiller supports the following distillation methods on CIFAR-100, ImageNet and MS-COCO:
|Method|Paper Link|CIFAR-100|ImageNet|MS-COCO|
|:---:|:---:|:---:|:---:|:---:|
|KD| <https://arxiv.org/abs/1503.02531> |&check;|&check;| |
|FitNet| <https://arxiv.org/abs/1412.6550> |&check;| | |
|AT| <https://arxiv.org/abs/1612.03928> |&check;|&check;| |
|NST| <https://arxiv.org/abs/1707.01219> |&check;| | |
|PKT| <https://arxiv.org/abs/1803.10837> |&check;| | |
|KDSVD| <https://arxiv.org/abs/1807.06819> |&check;| | |
|OFD| <https://arxiv.org/abs/1904.01866> |&check;|&check;| |
|RKD| <https://arxiv.org/abs/1904.05068> |&check;| | |
|VID| <https://arxiv.org/abs/1904.05835> |&check;| | |
|SP| <https://arxiv.org/abs/1907.09682> |&check;| | |
|CRD| <https://arxiv.org/abs/1910.10699> |&check;|&check;| |
|ReviewKD| <https://arxiv.org/abs/2104.09044> |&check;|&check;|&check;|
|DKD| <https://arxiv.org/abs/2203.08679> |&check;|&check;|&check;|


### Installation

Environments:

- Python 3.7
- PyTorch 1.9.0
- torchvision 0.10.0

Install the package:

```
sudo pip install -r requirements.txt
sudo python setup.py develop
```

### Getting started

0. Wandb as the logger

- The registeration: <https://wandb.ai/home>.
- If you don't want wandb as your logger, set `CFG.LOG.WANDB` as `False` at `mdistiller/engine/cfg.py`.

1. Evaluation

- You can evaluate the performance of our models or models trained by yourself.

- Our models are at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>, please download the checkpoints to `./download_ckpts`

- If test the models on ImageNet, please download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # evaluate teachers
  python3 tools/eval.py -m resnet32x4 # resnet32x4 on cifar100
  python3 tools/eval.py -m ResNet34 -d imagenet # ResNet34 on imagenet
  
  # evaluate students
  python3 tools/eval.py -m resnet8x4 -c download_ckpts/dkd_resnet8x4 # dkd-resnet8x4 on cifar100
  python3 tools/eval.py -m MobileNetV1 -c download_ckpts/imgnet_dkd_mv1 -d imagenet # dkd-mv1 on imagenet
  python3 tools/eval.py -m model_name -c output/your_exp/student_best # your checkpoints
  ```


2. Training on CIFAR-100

- Download the `cifar_teachers.tar` at <https://github.com/megvii-research/mdistiller/releases/tag/checkpoints> and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

  ```bash
  # for instance, DKD method.
  python tools/train.py --cfg configs/cifar100/dkd/resn56_res20.yaml
  
  # for instance, DCGD+KD method.
  python tools/train.py --cfg configs/cifar100/dkd/our_DCGD_DKD_res56_res20.yaml
  
  # for instance, DCGD+KD method.
  python tools/train.py --cfg configs/cifar100/kd/dcgd_kd_r56_r20.yaml
  ```


### Custom Distillation Method

1. create a python file at `mdistiller/distillers/` and define the distiller
  
  ```python
  from ._base import Distiller

  class MyDistiller(Distiller):
      def __init__(self, student, teacher, cfg):
          super(MyDistiller, self).__init__(student, teacher)
          self.hyper1 = cfg.MyDistiller.hyper1
          ...

      def forward_train(self, image, target, **kwargs):
          # return the output logits and a Dict of losses
          ...
      # rewrite the get_learnable_parameters function if there are more nn modules for distillation.
      # rewrite the get_extra_parameters if you want to obtain the extra cost.
    ...
  ```

2. regist the distiller in `distiller_dict` at `mdistiller/distillers/__init__.py`

3. regist the corresponding hyper-parameters at `mdistiller/engines/cfg.py`

4. create a new config file and test it.

# Citation

If this repo is helpful for your research, please consider citing the paper:

```BibTeX

```

# License

DCGD is released under the MIT license. See [LICENSE](LICENSE) for details.

# Acknowledgement

- Thanks for DKD. We build this library based on the [DKD's codebase](https://github.com/megvii-research/mdistiller).
