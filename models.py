from torch import nn 
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