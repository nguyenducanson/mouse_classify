import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# Constants and other configurations.
TEST_IMAGE = '../input/data/test'
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
NUM_WORKERS = 4
CLASS_NAMES = ['left', 'right']


# Define the model architecture.
class MedicalMNISTCNN(nn.Module):
    def __init__(self, num_classes=None):
        super(MedicalMNISTCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout2d(p=0.4),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.classifier(x)
        return x


def test_transform(IMAGE_RESIZE):
    transform = transforms.Compose([
            transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
                )
        ])
    return transform


def infer(model, image, DEVICE):
    model.eval()
    print('Infer image')

    image = image.to(DEVICE)
    # Forward pass.
    outputs = model(image)
    # Softmax probabilities.
    predictions = F.softmax(outputs).cpu().numpy()
    # Predicted class number.
    output_class = np.argmax(predictions)
    class_name = CLASS_NAMES[int(output_class)]
    return class_name


if __name__ == '__main__':
    image = Image.open(TEST_IMAGE)
    transform = test_transform(IMAGE_RESIZE)
    transformed_image = transform(image)

    checkpoint = torch.load('../outputs/model.pth')
    model = MedicalMNISTCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_name = infer(model, image, DEVICE)
    print('Class of image is:', class_name)
