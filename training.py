import os
import torch
import torchvision
import torchvision.transforms as tfms
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)



image_size = 224
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing



transforms = tfms.Compose([tfms.Resize((image_size, image_size)),
                           tfms.ToTensor(),
                           tfms.Normalize(imagenet_mean, imagenet_std)])

def load_model():
    model = vgg_model
    model.classifier[6] = nn.Linear(in_features=4096, out_features=40, bias=True).eval()
    return model

def train_loop(data_dir, weights_dir, epochs=100):

    train_dataset = torchvision.datasets.CelebA(data_dir, split="train", target_type=["attr"],
                                                transform=transforms)
    val_dataset = torchvision.datasets.CelebA(data_dir, split="valid", target_type=["attr"],
                                                transform=transforms)

    train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=32,
                              shuffle=False,
                              num_workers=4)
    model = load_model()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
    #acc=torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes).to(device)
    for epoch in tqdm(range(epochs)):

        model.train()
        epoch_train_loss, epoch_train_acc = [], []

        for i, data in enumerate(train_dataloader):
            if i>10:
                break
            inputs = data[0]
            labels = data[1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            outputs = (outputs > 0.5).float()
            curr_train_acc = np.mean(1-(labels-outputs).detach().numpy())

            # print statistics
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(curr_train_acc)
            if i % 1 ==0: # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{epochs},step: {i + 1:5d}/{len(train_dataloader)}] loss: {np.mean(epoch_train_loss):.3f}, acc: {np.mean(epoch_train_acc)}')
        scheduler.step()


        model.eval()
        epoch_val_loss, epoch_val_acc = [], []
        for i, data in enumerate(val_dataloader):
            if i>10:
                break
            inputs = data[0]
            labels = data[1]

            # forward
            outputs = model(inputs.float())
            #outputs = torch.nn.functional.softmax(outputs, dim=1)
            val_loss = criterion(outputs, labels.float())
            outputs = (outputs > 0.5).float()
            curr_val_acc = np.mean(1-(labels-outputs).detach().numpy())
            epoch_val_loss.append(val_loss.item())
            epoch_val_acc.append(curr_val_acc)


        torch.save(model, f'{weights_dir}/epoch_{epoch}_loss_{np.round(loss.detach().numpy(), decimals=3)}.pt')

        print(f'[epoch: {epoch + 1}/{epochs}] loss: {np.mean(epoch_val_loss):.3f}, acc: {np.mean(epoch_val_acc)}')
            # writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)
            # writer.add_scalar('Accuracy/val', np.mean(np.array(epoch_val_acc)), epoch)

if __name__ == '__main__':
    root_dir = r"C:\Users\shiri\Documents\School\Master\faces\CelebA\exps"
    data_dir =  r"C:\Users\shiri\Documents\School\Master\faces\CelebA"
    exp_name ='40_features_20_04'
    exp_dir = f'{root_dir}/{exp_name}'
    weights_dir = f'{exp_dir}/weights'
    for dir in [exp_dir, weights_dir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    train_loop(data_dir, weights_dir, 20)


