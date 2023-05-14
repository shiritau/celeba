import torch
import torchvision
import torchvision.transforms as tfms
import numpy as np


image_size = 224
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing



mytransforms = tfms.Compose([tfms.Resize((image_size, image_size)),
                            tfms.ToTensor(),
                           tfms.Normalize(imagenet_mean, imagenet_std)])

data_dir = r"C:\Users\shiri\Documents\School\Master\faces\CelebA"
train_dataset = torchvision.datasets.CelebA(data_dir, split="train", target_type=["attr"],transform=mytransforms)
feature_names= train_dataset.attr_names

weights_celeba = r"C:\Users\shiri\Documents\School\Master\faces\CelebA\exps\40_features_20_04\weights\epoch_8_loss_0.3499999940395355.pt"
model = torch.load(weights_celeba, map_location=torch.device('cpu'))
model.eval()
model.to("cpu")
img1_path= r"C:\Users\shiri\Documents\School\faces\Data\VGG-Face2\data\train\n004438\0015_01.jpg"
img = torchvision.io.read_image(img1_path)
#normalize = tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#composed_transforms = tfms.Compose([Rescale((224, 224)), normalize])
#image = composed_transforms(img)
image = mytransforms(img.float())
features = model(image.unsqueeze(0).float())

# get feature index where value is greater than 0
feature_idx = np.where(features.detach().numpy() > 0)[1]
features_output = []
for idx in feature_idx:
    features_output.append(feature_names[idx])

# convert to string
features_output = ', '.join(features_output)
print(features_output)

