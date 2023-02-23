import os
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


# Set the path to the dataset folder
dataset_path = r'C:\Users\ra_saval\Desktop\fulhas\Data for test'

# Set the train/test/validation split ratios
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Get the list of classes
class_names = sorted(os.listdir(dataset_path))

for split in ['train', 'test', 'val']:
    for class_name in class_names:
        os.makedirs(os.path.join(split, class_name), exist_ok=True)

# Split the dataset into train and test sets
for class_name in class_names:
    images = os.listdir(os.path.join(dataset_path, class_name))
    train_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)
    for image in train_images:
        src = os.path.join(dataset_path, class_name, image)
        dst = os.path.join('train', class_name, image)
        shutil.copy(src, dst)
    for image in test_images:
        src = os.path.join(dataset_path, class_name, image)
        dst = os.path.join('test', class_name, image)
        shutil.copy(src, dst)

# Split the training set into train and validation sets
for class_name in class_names:
    train_images = os.listdir(os.path.join('train', class_name))
    train_images, val_images = train_test_split(train_images, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)
    for image in val_images:
        src = os.path.join('train', class_name, image)
        dst = os.path.join('val', class_name, image)
        shutil.move(src, dst)
