from torch import nn 
import timm

class Network(nn.Module):
    def __init__(self, emb_dim=256):
        super(Network, self).__init__()

        self.base = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=100)
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        
    def forward(self, x):
        x = self.base.forward_features(x)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x
    
    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True