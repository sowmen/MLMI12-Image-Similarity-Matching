import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn import metrics
import gc
import math

import timm
from torchmetrics import Accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb


torch.backends.cudnn.benchmark = True
from dataset import TinyImagenet
from utils import *

NUM_CLASSES = 100
OUTPUT_DIR = ""
device =  'cuda'
config_defaults = {
    "epochs": 2,
    "train_batch_size": 2,
    "valid_batch_size": 4,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    # "weight_decay": 0.0001,
    # "schedule_patience": 5,
    # "schedule_factor": 0.25,
    "model": "EffNetB6",
}

def train(name, train_df, val_df, resume=None):
    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)

    os.makedirs(f'{OUTPUT_DIR}/weights', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/checkpoint', exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    wandb.init(project="MLMI12", entity="sdipto", config=config_defaults, name=run)
    config = wandb.config


    model = timm.create_model('tf_efficientnet_b6_ns', pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    # for name_, param in model.named_parameters():
    #     if 'classifier' in name_:
    #         continue
    #     else:
    #         param.requires_grad = False    
    

    ########################-- CREATE DATASET and DATALOADER --########################
    train_dataset = TinyImagenet(train_df, mode="train", triplet=False)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    valid_dataset = TinyImagenet(val_df, mode="valid", triplet=False)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
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

        print(f"TRAIN_LOSS = {train_metrics['train_loss']}  |  TRAIN_ACC = {train_metrics['train_acc']}  |  TRAIN_ACC_TOP5 = {train_metrics['train_acc_top5']}")
        print(f"VALID_LOSS = {valid_metrics['valid_loss']}  |  VALID_ACC = {valid_metrics['valid_acc']}  |  VALID_ACC_TOP5 = {valid_metrics['valid_acc_top5']}")
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


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()

    total_loss = AverageMeter()
    total_acc = AverageMeter()
    total_acc_top5 = AverageMeter()

    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    accuracy5 = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=5)

    for images, labels in tqdm(train_loader):
        images = images.to(device)

        optimizer.zero_grad()

        preds = model(images)
        print(labels.shape, preds.shape)
        loss = criterion(preds, labels.to(device))
        loss.backward()

        optimizer.step()
        
        #---------------------Batch Loss Update-------------------------
        total_loss.update(loss.item(), train_loader.batch_size)
        total_acc.update(accuracy(preds.cpu(), labels.cpu()).item(), train_loader.batch_size)
        total_acc.update(accuracy5(preds.cpu(), labels.cpu()).item(), train_loader.batch_size)

        
    train_metrics = {
        "train_loss" : total_loss.avg,
        "train_acc" : total_acc.avg,
        "train_acc_top5" : total_acc_top5.avg,
        "epoch" : epoch,
        "train_learning_rate" : optimizer.param_groups[0]['lr']
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    total_loss = AverageMeter()
    total_acc = AverageMeter()
    total_acc_top5 = AverageMeter()

    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    accuracy5 = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=5)

    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            loss = criterion(preds, labels)
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), valid_loader.batch_size)
            total_acc.update(accuracy(preds.cpu(), labels.cpu()).item(), valid_loader.batch_size)
            total_acc.update(accuracy5(preds.cpu(), labels.cpu()).item(), valid_loader.batch_size)

        
    valid_metrics = {
        "valid_loss" : total_loss.avg,
        "valid_acc" : total_acc.avg,
        "valid_acc_top5" : total_acc_top5.avg,
        "epoch" : epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics


if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    train(
        name=f"MultiClass" + config_defaults["model"],
        train_df=train_df,
        val_df=val_df,
        resume=None
    )