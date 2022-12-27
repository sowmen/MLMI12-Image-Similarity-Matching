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
from models import Network

NUM_CLASSES = 100
OUTPUT_DIR = "/content/drive/MyDrive/MLMI12-Project"
device =  'cuda'
config_defaults = {
    "epochs": 40,
    "train_batch_size": 64,
    "valid_batch_size": 32,
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


    model = Network()
    model.to(device)

    # for name_, param in model.named_parameters():
    #     if 'classifier' in name_:
    #         continue
    #     else:
    #         param.requires_grad = False    
    

    ########################-- CREATE DATASET and DATALOADER --########################
    train_dataset = TinyImagenet(train_df, mode="train", triplet=True)
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    valid_dataset = TinyImagenet(val_df, mode="valid", triplet=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.TripletMarginLoss().to(device)
    es = EarlyStopping(patience=10, mode="min")


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

        print(f"TRAIN_LOSS = {train_metrics['train_loss']}")
        print(f"VALID_LOSS = {valid_metrics['valid_loss']}")
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

    for anchor, positive, negative, _ in tqdm(train_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()

        optimizer.step()
        
        #---------------------Batch Loss Update-------------------------
        total_loss.update(loss.item(), train_loader.batch_size)

        
    train_metrics = {
        "train_loss" : total_loss.avg,
        "epoch" : epoch,
        "train_learning_rate" : optimizer.param_groups[0]['lr']
    }
    wandb.log(train_metrics)

    return train_metrics


def valid_epoch(model, valid_loader, criterion, epoch):
    model.eval()

    total_loss = AverageMeter()

    with torch.no_grad():
        for anchor, positive, negative, _ in tqdm(valid_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            
            #---------------------Batch Loss Update-------------------------
            total_loss.update(loss.item(), valid_loader.batch_size)

        
    valid_metrics = {
        "valid_loss" : total_loss.avg,
        "epoch" : epoch
    }
    wandb.log(valid_metrics)

    return valid_metrics


if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    train(
        name=f"Triplet512" + config_defaults["model"],
        train_df=train_df,
        val_df=val_df,
        resume=None
    )