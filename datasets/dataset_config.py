import os
from dataclasses import dataclass

@dataclass
class DatasetConfig():
    num_classes: int = 10
    min_instances_num: int = 199
    total_set_size:int  = 60
    train_set_size: int = 50
    test_set_size: int  = 10
    plot: bool = False
    model_type: str = 'resnet_pretrained_mtcnn' # 'resnet_pretrained_mtcnn' or 'pretrained_celeba'
    pretrained_celeba_weights: str = r"C:\Users\shiri\Documents\School\Master\faces\CelebA\exps\40_features_20_04\weights\epoch_8_loss_0.3499999940395355.pt"
    resnet_pretrained_mtcnn_weights: str = r"C:\Users\shiri\Documents\School\Galit\epoch_60_weights.pt"

    #dbscan params:
    pretrained_celeba_eps_idx: float = 100
    pretrained_celeba_min_samples: int = 5
    resnet_pretrained_mtcnn_eps_idx: float = 100
    resnet_pretrained_mtcnn_min_samples: int = 5