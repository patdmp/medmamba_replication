import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import json

def load_data(dataset_name, batch_size=32):
    """
    Function to load MedMNIST dataset (e.g., RetinaMNIST).
    
    :param dataset_name: str, name of the dataset (e.g., 'retinamnist')
    :param batch_size: int, batch size for data loaders
    :return: train_loader, val_loader, train_num, val_num
    """
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # Define transformations
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.Lambda(lambda image: image.convert('RGB')),
                                     transforms.AugMix(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[.5], std=[.5])
        ]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.Lambda(lambda image: image.convert('RGB')),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[.5], std=[.5])
        ])
    }

    # Load datasets
    train_dataset = DataClass(split='train', download=True, transform=data_transform['train'])
    val_dataset = DataClass(split='val', download=True, transform=data_transform['val'])

    # Number of samples
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get class labels from MedMNIST info
    info = INFO[dataset_name]
    classes = info['label']
    
    # Map indices to class names
    idx_to_class = {i: label for i, label in enumerate(classes)}

    # Save the mapping to a JSON file
    json_label_path = f'{dataset_name}/class_indices.json'
    with open(json_label_path, 'w') as json_file:
        json.dump(idx_to_class, json_file, indent=4)

    return train_loader, val_loader, train_num, val_num
