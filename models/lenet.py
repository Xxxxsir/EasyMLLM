import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # the input dim of the fc layer :12x12x64
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def lenet(num_classes:int = 10,pretrained:bool = False,pretrained_path:str = None,**kwargs):

    model = LeNet(
        num_classes=num_classes,
    )

    print(f"ViT Model Created with hyperparameters:\n"
          f"  num_classes = {num_classes}\n")

    if pretrained:
        if pretrained_path is None:
            raise ValueError("pretrained_path must be specified when pretrained is True")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Load pretrained Lenet model Successfully from {pretrained_path}")

    return model