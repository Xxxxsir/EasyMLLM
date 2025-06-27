import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path
from PIL import Image
from torch.utils.data import random_split


idx_to_class_covid = {
    0:"COVID",
    1:"Lung_Opacity",
    2:"Normal",
    3:"Viral Pneumonia",
}

idx_to_class_mnist = {i: str(i) for i in range(10)}

class CovidRadioGraphyDataset(Dataset):
    def __init__(self,root_dir:str,transform=None):

        self.transform = transform
        self.samples = []

        class_names = [item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))]
        self.class_to_idx = {class_name:idx for idx,class_name in enumerate(class_names)}

        for class_name in class_names:
            img_dir = os.path.join(root_dir,class_name,"images")
            for img in os.listdir(img_dir):
                if img.endswith('.png'):
                    img_path = os.path.join(img_dir, img)
                    img_label = self.class_to_idx[class_name]
                    self.samples.append((img_path, img_label))
        
    def __len__(self):
        return len(self.samples)
    
    def load_image(self, image_path):
        return Image.open(image_path)

    def __getitem__(self, index):
        img_path,label = self.samples[index]
        image = self.load_image(img_path).convert("RGB")

        if self.transform:
            return self.transform(image), label
        else:
            return image, label

    
def get_transform_covidradiography(split:str = 'train'):
    if split == 'train':
        return transforms.Compose(
            [
                transforms.RandomRotation(90),
                transforms.Resize((256,256)),
                transforms.RandomCrop((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.3738, 0.3738, 0.3738),
                                    (0.3240, 0.3240, 0.3240))
            ]
        )
    elif split == 'val' or split == 'test':
        return transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.3738, 0.3738, 0.3738),
                                    (0.3240, 0.3240, 0.3240))
            ]
        )

def get_transform(dataset_name:str, split:str = 'train'):
    if dataset_name == "covid":
        return get_transform_covidradiography(split)
    elif dataset_name == "mnist":
        if split == 'train':
            return transforms.Compose(
                [
                    transforms.RandomRotation(10),
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )
        elif split == 'test':
            return transforms.Compose(
                [
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported now.")

def get_dataloaders(dataset_name,dataset_root_dir,batch_size,num_workers=4, pin_flag:bool = True):
    root_dir = Path(dataset_root_dir)
    if dataset_name == "covid":
        train_dataset = CovidRadioGraphyDataset(root_dir=root_dir / "train", transform=get_transform_covidradiography("train"))
        val_dataset   = CovidRadioGraphyDataset(root_dir=root_dir / "val",   transform=get_transform_covidradiography("val"))
        test_dataset  = CovidRadioGraphyDataset(root_dir=root_dir / "test",  transform=get_transform_covidradiography("test"))
    elif dataset_name == "mnist":
        origin_train_dataset = datasets.MNIST(root=dataset_root_dir,train=True,download=True,transform=get_transform("mnist", "train"))
        # Split the MNIST dataset into train and validation sets
        train_size = 50000
        val_size   = 10000
        train_dataset, val_dataset = random_split(origin_train_dataset, [train_size, val_size])
        test_dataset  = datasets.MNIST(root=dataset_root_dir,train=False,download=True,transform=get_transform("mnist", "test"))
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported now.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True  ,pin_memory=pin_flag)
    val_loader   = DataLoader(val_dataset  , batch_size=batch_size, num_workers=num_workers, shuffle=False ,pin_memory=pin_flag)
    test_loader  = DataLoader(test_dataset , batch_size=batch_size, num_workers=num_workers, shuffle=False ,pin_memory=pin_flag)

    return train_loader, val_loader, test_loader