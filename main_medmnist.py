import os
import sys
import json

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
from tqdm import tqdm
from data_loader import load_data
from medmamba import VSSM, medmamba_t, medmamba_s, medmamba_b
from collections import Counter
from MedViT import MedViT_small

def main():
    # Set up device: use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # Load dataset
    dataset_name = 'breastmnist'
    num_classes = 2

    # Check if the folder exists, if not, create it
    if not os.path.exists(f'{dataset_name}/'):
        os.makedirs(f'{dataset_name}/')

    batch_size = 64
    train_loader, val_loader, train_num, val_num = load_data(dataset_name, batch_size)
    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

    def count_labels(train_loader):
        # Initialize a counter to store the label counts
        label_counter = Counter()

        # Iterate through the training data loader
        for images, labels in train_loader:
            # Flatten labels (in case they are not already flattened)
            labels = labels.view(-1)
            # Update the counter with the labels in this batch
            label_counter.update(labels.tolist())

        return label_counter

    # Example usage
    label_counts = count_labels(train_loader)
    print(f"Label counts: {label_counts}")

    # Initialize the model
    medmamba_s = VSSM(depths=[2, 2, 8, 2],dims=[96,192,384,768],num_classes=num_classes)
    medmamba_b = VSSM(depths=[2, 2, 12, 2],dims=[128,256,512,1024],num_classes=num_classes)
    medmamba_t = VSSM(depths=[2, 2, 4, 2],dims=[96,192,384,768],num_classes=num_classes)
    model_name = 'medmamba_t'
    net = medmamba_t
    net.num_classes =num_classes
    net.to(device)

    # Loss and optimizer
    # class_weights = torch.tensor([1.0, 399/147], device=device)  # Adjust weights based on class distribution
    # loss_function = nn.CrossEntropyLoss(weight=class_weights)

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001)
    # optimizer = SGD(net.parameters(), lr=0.001)

    # Set up the learning rate scheduler
    milestones = [50, 75] #range(10,100,10)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # Set training parameters
    epochs = 10
    best_acc = 0.0
    train_steps = len(train_loader)
    save_path = f'{dataset_name}/{model_name}Net.pth'

    for epoch in range(epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
        
        scheduler.step()

        # Validation phase
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        
        # Save the model if the validation accuracy improves
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Training Done!')


if __name__ == '__main__':
    main()
