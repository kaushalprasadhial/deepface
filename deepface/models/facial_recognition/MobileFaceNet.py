import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

WEIGHTS_URL = "https://drive.google.com/uc?id=1oLc8SgBFDOrvg-Lt-bS-_2UrHx7eSX7o"  # Replace with actual URL 1oLc8SgBFDOrvg-Lt-bS-_2UrHx7eSX7o

# Define MobileFaceNet Architecture
class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super(MobileFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.prelu2 = nn.PReLU(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.prelu3 = nn.PReLU(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.prelu4 = nn.PReLU(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.prelu4(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

# Define load_model function
def load_model(url):
    """
    Load the MobileFaceNet model with pre-trained weights.
    
    Args:
        url (str): URL of the model weights.
    
    Returns:
        torch.nn.Module: Loaded MobileFaceNet model.
    """
    model = MobileFaceNet()
    state_dict = load_state_dict_from_url(url, progress=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Define MobileFaceNetClient class
class MobileFaceNetClient:
    """
    MobileFaceNet model client for face recognition.
    """
    def __init__(self):
        self.model = load_model(self.WEIGHTS_URL)
        self.model_name = "MobileFaceNet"
        self.input_shape = (3, 112, 112)  # RGB image of size 112x112
        self.output_shape = 512  # Feature embedding size

