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
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from torchvision import transforms
from sklearn.metrics import accuracy_score


torch.backends.cudnn.benchmark = True
from dataset import TinyImagenet
from utils import *
from models import Network

NUM_CLASSES = 50
OUTPUT_DIR = "/content/drive/MyDrive/MLMI12-Project"
device =  'cuda'
config_defaults = {
    "epochs": 40,
    "train_batch_size": 60,
    "valid_batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    # "weight_decay": 0.0001,
    # "schedule_patience": 5,
    # "schedule_factor": 0.25,
    "model": "end-end50",
}

def train(name, train_df, val_df, resume=None):
    dt_string = datetime.now().strftime("%d|%m_%H|%M|%S")
    print("Starting -->", dt_string)

    os.makedirs(f'{OUTPUT_DIR}/weights', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/checkpoint', exist_ok=True)
    run = f"{name}_[{dt_string}]"
    
    wandb.init(project="MLMI12", entity="sdipto", config=config_defaults, name=run)
    config = wandb.config


    model = Network(num_classes = NUM_CLASSES, emb_dim=128)
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
    es = EarlyStopping(patience=7, mode="min")


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

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "weights", f"{run}.h5")))
    test(model)


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

        similarity_outs = []
        labels = []
        for row in tqdm(seen_seen_df):

            im1 = cv2.imread(os.path.join('tiny-imagenet-200', row[0]), cv2.IMREAD_COLOR)
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            tensor1 = transform(im1).unsqueeze(0).to(device)

            feature1 = model(tensor1).cpu()
            # feature1 = torch.nn.functional.normalize(feature1)


            ########################################

            im2 = cv2.imread(os.path.join('tiny-imagenet-200', row[1]), cv2.IMREAD_COLOR)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            tensor2 = transform(im2).unsqueeze(0).to(device)

            feature2 = model(tensor2).cpu()
            # feature2 = torch.nn.functional.normalize(feature2)

            #########################################

            labels.append(row[2])

            csim = torch.nn.functional.cosine_similarity(feature1.squeeze(), feature2.squeeze(), dim=0).item()
            similarity_outs.append(1 if csim > 0.1 else 0)

        csim_acc = accuracy_score(labels, similarity_outs)
        print("Cosine Similarity (0.1) Accuracy: ", csim_acc)

        wandb.log({
            f"{path.split('.')[0]}": csim_acc
        })

    
    test_df('seen_seen_test.csv')
    test_df('seen_unseen_test.csv')
    test_df('unseen_unseen_test.csv')


if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('val.csv')

    train(
        name=f"Triplet128" + config_defaults["model"],
        train_df=train_df,
        val_df=val_df,
        resume=None
    )