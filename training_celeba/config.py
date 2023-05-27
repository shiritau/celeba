from dataclasses import dataclass

@dataclass
class trainConfig():
    root_dir: str = r"C:\Users\shiri\Documents\School\Master\faces\CelebA\exps"
    data_dir: str = r"C:\Users\shiri\Documents\School\Master\faces\CelebA"
    exp_name: str  = '40_features_20_04'
    epochs: int = 10
