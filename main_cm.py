import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from data_loader import load_data
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from medmamba import VSSM, medmamba_t, medmamba_s, medmamba_b

class ConfusionMatrix:
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("The model accuracy is ", acc)

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self, dataset_name='', save_path=None):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # Set x-axis and y-axis labels
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        
        # Show colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # Add labels on the matrix
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                        verticalalignment='center',
                        horizontalalignment='center',
                        color="white" if info > thresh else "black")
        
        plt.tight_layout()

        # Save the plot as an image file
        if save_path is None:
            save_path = f'{dataset_name}_confusion_matrix.png'  # Default file name
        plt.savefig(save_path)
        print(f"Confusion matrix saved as {save_path}")



def main():
    # Set up device: use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # Load dataset
    dataset_name = 'retinamnist'
    num_classes = 5

    batch_size = 128
    train_loader, val_loader, train_num, val_num = load_data(dataset_name, batch_size)
    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

    # Initialize the model
    model_name = 'medmamba_t'
    net = medmamba_s
    net.num_classes = num_classes
    net.to(device)

    # Load trained model weights
    model_weight_path = f'{dataset_name}/{model_name}Net.pth'
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # Prepare labels for confusion matrix
    json_label_path = f'{dataset_name}/class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    with open(json_label_path, 'r') as json_file:
        class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)

    # Evaluate the model and update the confusion matrix
    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            confusion.update(predict_y.to("cpu").numpy(), val_labels.to("cpu").numpy())
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / val_num
    print('Validation Accuracy: {:.3f}'.format(val_accurate))

    # Print confusion matrix summary and plot
    confusion.summary()
    save_path = f'ConfusionMatrix/confusion_matrix_{dataset_name}_{model_name}.png'
    confusion.plot(dataset_name=dataset_name, save_path=save_path)


if __name__ == '__main__':
    main()
