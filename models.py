from torch import nn 
import torch
import timm

class Network(nn.Module):
    def __init__(self, num_classes, emb_dim):
        super(Network, self).__init__()

        self.base = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=num_classes)
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        
    def forward(self, x):
        x = self.base.forward_features(x)
        x = self.projection(x)
        x = nn.functional.normalize(x)
        return x
    
    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True


class ComboNet(nn.Module):
    def __init__(self, num_classes):
        super(ComboNet, self).__init__()

        self.base = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=num_classes)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(3584, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        x1 = self.base.forward_features(x1)
        x2 = self.base.forward_features(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.classifier(x)
        return x
    
    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True