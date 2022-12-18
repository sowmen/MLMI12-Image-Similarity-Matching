
import PIL
import numpy as np
import cv2
import random
import os
from torchvision import transforms
from torch.utils.data import Dataset
from utils import label_to_id

class TinyImagenet(Dataset):
    def __init__(self, df, mode='train', triplet=True):
        self.DATA_ROOT = 'tiny-imagenet-200'
        self.mode = mode
        self.triplet = triplet
                   
        self.images = df.image_path.values
        self.labels = df.label.values
        self.index = df.index.values

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, item):
        anchor_path = self.images[item]
        anchor_image = cv2.imread(os.path.join(self.DATA_ROOT, anchor_path), cv2.IMREAD_COLOR)
        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
        
        anchor_label = self.labels[item]

        if self.transform:
            anchor_image = self.transform(anchor_image)

        if self.triplet:
            positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]
            positive_path = self.images[random.choice(positive_list)]
            positive_image = cv2.imread(os.path.join(self.DATA_ROOT, positive_path), cv2.IMREAD_COLOR)
            positive_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2RGB)
            
            negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
            negative_path = self.images[random.choice(negative_list)]
            negative_image = cv2.imread(os.path.join(self.DATA_ROOT, negative_path), cv2.IMREAD_COLOR)
            negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)

            return anchor_image, positive_image, negative_image, label_to_id(anchor_label)
        
        else:
            return anchor_image, label_to_id(anchor_label)