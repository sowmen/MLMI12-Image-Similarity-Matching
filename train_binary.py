import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn import metrics
import gc
import math
import cv2
import timm
from torchmetrics import Accuracy

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import accuracy_score


torch.backends.cudnn.benchmark = True
from dataset import ComboDataset
from utils import *
from models import ComboNet

NUM_CLASSES = 50
OUTPUT_DIR = "/content/drive/MyDrive/MLMI12-Project"
device =  'cuda'
config_defaults = {
    "epochs": 40,
    "train_batch_size": 80,
    "valid_batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    # "weight_decay": 0.0001,
    # "schedule_patience": 5,
    # "schedule_factor": 0.25,
    "model": "EffNetB4",
}

def train(name, train_df, val_df, resume=None):
    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)

    os.makedirs(f'{OUTPUT_DIR}/weights', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/checkpoint', exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    wandb.init(project="MLMI12", entity="sdipto", config=config_defaults, name=run)
    config = wandb.config


    model = ComboNet(NUM_CLASSES)
    model.to(device)

    # for name_, param in model.named_parameters():
    #     if 'classifier' in name_:
    #         continue
    #     else:
    #         param.requires_grad = False    
    

    ########################-- CREATE DATASET and DATALOADER --########################
    train_dataset = ComboDataset(train_df, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    valid_dataset = ComboDataset(val_df, mode="valid")
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss().to(device)
    es = EarlyStopping(patience=5, mode="min")


    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("-----------> Resuming <------------")

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch = {epoch}/{config.epochs-1}")
        print("------------------")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch)
        valid_metrics = valid_epoch(model, valid_loader, criterion, epoch)

        print(f"TRAIN_LOSS = {train_metrics['train_loss']}  |  TRAIN_ACC = {train_metrics['train_acc']}")
        print(f"VALID_LOSS = {valid_metrics['valid_loss']}  |  VALID_ACC = {valid_metrics['valid_acc']}")
        print("New LR", optimizer.param_groups[0]['lr'])

        
        es(
            valid_metrics['valid_loss'],
            model,
            model_path=os.path.join(OUTPUT_DIR, "weights", f"{run}.h5"),
        )
        if es.early_stop:
            print("Early stopping")
            break

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, 'checkpoint', f"{run}.pt"))

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "weights", f"{run}.h5")))
    test(model)


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    total_loss = AverageMeter()
    total_acc = AverageMeter()

    accuracy = Accuracy(task="binary", num_classes=NUM_CLASSES)

    for image1, image2, labels in tqdm(train_loader):
        image1 = image1.to(device)
        image2 = image2.to(device)

        optimizer.zero_grad()

        preds = model(image1, image2).squeeze()
        loss = criterion(preds, labels.to(device))
        loss.backward()

        optimizer.step()
        
        #---------------------Batch Loss Update-------------------------
        total_loss.update(loss.item(), train_loader.batch_size)
        total_acc.update(accuracy(preds.cpu(), (labels.cpu() > 0.5).to(torch.int32)).item(), train_loader.batch_size)

        
    train_metrics = {
        "train_loss" : total_loss.avg,
        "train_acc" : total_acc.avg * 100,
        "epoch" : epoch,
        "train_learning_rate" : optimizer.param_groups[0]['lr']
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    total_loss = AverageMeter()
    total_acc = AverageMeter()

    accuracy = Accuracy(task="binary", num_classes=NUM_CLASSES)

    with torch.no_grad():
        for image1, image2, labels in tqdm(valid_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)

            preds = model(image1, image2).squeeze()

            loss = criterion(preds, labels.to(device))
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), valid_loader.batch_size)
            total_acc.update(accuracy(preds.cpu(), (labels.cpu() > 0.5).to(torch.int32)).item(), valid_loader.batch_size)

        
    valid_metrics = {
        "valid_loss" : total_loss.avg,
        "valid_acc" : total_acc.avg * 100,
        "epoch" : epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics


def test(model):
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def test_df(path): 
        print("Testing ---- ", path)

        seen_seen_df = pd.read_csv(path).values

        pred_outs = []
        labels = []
        for row in tqdm(seen_seen_df):

            im1 = cv2.imread(os.path.join('tiny-imagenet-200', row[0]), cv2.IMREAD_COLOR)
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            tensor1 = transform(im1).unsqueeze(0).to(device)

            ########################################

            im2 = cv2.imread(os.path.join('tiny-imagenet-200', row[1]), cv2.IMREAD_COLOR)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            tensor2 = transform(im2).unsqueeze(0).to(device)

            out = model(tensor1, tensor2).cpu().squeeze()

            #########################################

            pred_outs.append(0 if out.item() < 0.5 else 1)
            labels.append(row[2])

        
        b_acc = accuracy_score(labels, pred_outs)

        print("Binary Accuracy: ", b_acc)

        wandb.log({
            f"{path.split('.')[0]}_binary_acc": b_acc,
        })

    
    test_df('seen_seen_test.csv')
    test_df('seen_unseen_test.csv')
    test_df('unseen_unseen_test.csv')


if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    train(
        name=f"Binary" + config_defaults["model"],
        train_df=train_df,
        val_df=val_df,
        resume=None
    )