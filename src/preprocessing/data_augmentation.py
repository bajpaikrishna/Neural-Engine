import torchvision.transforms as transforms

def get_data_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])

def apply_augmentation(dataset, transforms):
    dataset.transform = transforms
    return dataset
