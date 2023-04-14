import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# Define transforms for the data
transform = transforms.Compose([    
    transforms.Resize(224),    
    transforms.CenterCrop(224),    
    transforms.ToTensor(),
    transforms.Normalize([0.4693, 0.4424, 0.3985], [0.2367, 0.2270, 0.2310])
])

batch_size = 32
# Load the data
dataset = datasets.ImageFolder('Stanford40/Images', transform=transform)

# Determine the size of the train and test sets
train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
valid_size = len(dataset) - train_size - test_size

# Split the dataset into training and testing sets
train_dataset, test_dataset, valid_dataset = random_split(dataset, [train_size, test_size, valid_size])

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

print('train size %d  test size %d' % (train_size, test_size))

# Define the ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, num_classes=12):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResNetBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNetBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Initialize the model and define the optimizer and loss function
device = torch.device("mps")
model = ResNet(num_classes=12).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print(model)

num_epochs = 10

for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    
    # training
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    for data, target in valid_loader:
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        valid_loss += loss.item() * data.size(0)
    
    train_loss /= len(train_loader.dataset)
    valid_loss /= len(valid_loader.dataset)
    
    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} Accuracy: {}'.format(
        epoch+1, train_loss, valid_loss, correct / total))


