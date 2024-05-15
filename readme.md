# SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks

## Installing Dependencies

```bash
pip3 install torch torchvision
pip3 install tensorboard thop spikingjelly==0.0.0.0.14 cupy-cuda11x timm
```

## Usage

### Experiments on ImageNet

To reproduce the experiments on ImageNet in the paper, you need to first organize the dataset as follows

```bash
/path/to/your/dataset
|-- train
|   |-- n01440764
|   |-- n01443537
|   `-- ...
`-- val
    |-- n01440764
    |-- n01443537
    `-- ...
```

Then run the following command to reproduce the experiment of SpikingResformer-S

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
main.py \
    -c configs/main/spikingresformer_s.yaml \
    --data-path /path/to/your/dataset \
    --output-dir /path/to/your/output \
    ;
```

Experimental setups of SpikingResformer-Ti, M, L can be found in `configs/main`.

### Transfer Learning

Run the following command to transfer the pretrained SpikingResformer-S to CIFAR10

```bash
python \
main.py \
    -c configs/transfer/cifar10.yaml \
    --data-path /path/to/your/dataset \
    --output-dir /path/to/your/output \
    --transfer /path/to/your/checkpoint \
    ;
```

Experimental setups on other datasets can be found in `configs/transfer`.

### Direct Training

Run the following command to directly train SpikingResformer-Ti* on CIFAR10

```bash
python \
main.py \
    -c configs/direct_training/cifar10.yaml \
    --data-path /path/to/your/dataset \
    --output-dir /path/to/your/output \
    ;
```

Experimental setups on other datasets can be found in `configs/direct_training`.

## Citation

```bibtex
@inproceedings{shi2024spikingresformer,
    title={SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks}, 
    author={Shi, Xinyu and Hao, Zecheng and Yu, Zhaofei},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```
