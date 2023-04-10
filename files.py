from collections import Counter
import shutil
from sklearn.model_selection import train_test_split

keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike", "riding_a_horse", 
        "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]
with open('Stanford40/ImageSplits/train.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

with open('Stanford40/ImageSplits/test.txt', 'r') as f:
    # We won't use these splits but split them ourselves
    test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

# Combine the splits and split for keeping more images in the training set than the test set.
all_files = train_files + test_files
all_labels = train_labels + test_labels

for i in range(len(all_files)):
    shutil.move(f'Stanford40/JPEGImages/{all_files[i]}', f'Stanford40/Images/{all_labels[i]}')

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# Define transforms for the data
transform = transforms.Compose([    
    transforms.Resize(224),    
    transforms.CenterCrop(224),    
    transforms.ToTensor()
])

batch_size = 32
# Load the data
dataset = datasets.ImageFolder('Stanford40/Images', transform=transform)

loader = DataLoader(dataset, batch_size, shuffle=False)

# Compute the mean and standard deviation
mean = 0.
std = 0.
total = 0.

for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    total += batch_samples

mean /= total
std /= total