import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from image_model import CNNExtractor, HDCImageClassifier
from main import HDCPruner

data_dir = "data"
cnn_path = "models/cnn_fmnist.pth"
hdc_path = "models/hdc_fmnist.pth"
new_hdc_path = "models/hdc_fmnist_pruned.pth"

def train_fmnist():
    learning_rate = 0.001
    epochs = 10
    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset: Dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    cnn = CNNExtractor(input_channels=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    print("==TRAINING CNN on FashionMNIST==")
    for epoch in range(epochs): # train CNN
        total_loss = 0
        num_batches = 0
        for i, (images, labels) in enumerate(train_loader):
            pred = cnn(images)
            loss = criterion(pred, labels)

            total_loss += loss.item()
            num_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Average Loss={total_loss/num_batches}")

    torch.save(cnn.state_dict(), cnn_path)

    old_dim = 5000
    hdc = HDCImageClassifier(input_channels=1, hd_dim=old_dim)
    hdc.init_cnn(cnn_path)

    accs = hdc.train_hdc_iterative(train_loader)

    torch.save(hdc.state_dict(), hdc_path)

    pruner = HDCPruner(hdc)
    new_dim, proj = pruner.hd_prune(train_dataset)

    hdc = HDCImageClassifier(input_channels=1, hd_dim=new_dim)
    hdc.init_cnn(cnn_path)

    print(f"Evaluating on validation data with similarity threshold {0}")
    evaluation_results = hdc.evaluate(val_loader)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.2f}%")
    print(f"Valid Accuracy: {evaluation_results['valid_accuracy']:.2f}%")
    print(f"Average Similarity: {evaluation_results['average_similarity']:.4f}")
    print(f"Average Confidence: {evaluation_results['average_confidence']:.4f}")
    confusion = evaluation_results['confusion_matrix']
    per_class_acc = confusion.diagonal() / confusion.sum(axis=1) * 100
    print("\nPer-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc:.2f}%")