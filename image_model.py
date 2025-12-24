from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from main import HDCClassifier

class CNNExtractor(nn.Module):
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.flatten = nn.Flatten()
        self.projection = nn.Linear(128 * 4 * 4, feature_dim)

    def forward(self, x, return_features=True): # return_features is there for a placeholder
        x = self.conv_layers(x)
        x = self.flatten(x)
        features = self.projection(x)

        features = F.normalize(features, p=2, dim=1)
        
        return features

class HDCImageClassifier(nn.Module):
    def __init__(self, input_channels: int = 3, feature_dim: int = 512, hd_dim: int = 5000, n_classes: int = 10):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.hd_dim = hd_dim
        self.n_classes = n_classes
        self.feature_extractor = CNNExtractor(
            input_channels=input_channels,
            feature_dim=feature_dim,
        ).to(self.device)
        self.hdc = HDCClassifier(
            dimension=hd_dim,
            n_classes=n_classes
        ).to(self.device)

    def init_cnn(self, model_path: str):
        self.feature_extractor.load_state_dict(torch.load(model_path, map_location=self.device))
        self.feature_extractor.eval()
        print(f"Loaded CNN from {model_path}")
    
    def train_hdc_iterative(self, train_loader: torch.utils.data.DataLoader, n_iters: int = 3):
        """Retrains the model for n_iters"""
        self.feature_extractor.eval()
        
        all_features, all_labels = [], []
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                features = self.feature_extractor(images)
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        features_tensor = torch.cat(all_features, dim=0).to(self.device)
        labels_tensor = torch.cat(all_labels, dim=0).to(self.device)

        hypervectors = self.hdc.features_to_hypervector(features_tensor)
        hypervectors_signed = hypervectors.int() * 2 - 1

        self.hdc.prototype_accum.zero_()
        self.hdc.class_counts.zero_()
        for i in range(len(labels_tensor)):
            label = labels_tensor[i].item()
            self.hdc.prototype_accum[label] += hypervectors_signed[i]
            self.hdc.class_counts[label] += 1
        self.hdc.finalize_prototypes()

        accuracies = []
        for it in range(n_iters):
            predictions, _ = self.hdc.predict(features_tensor)

            misclassified_mask = predictions != labels_tensor

            for i in range(len(labels_tensor)):
                if misclassified_mask[i]:
                    true_label = labels_tensor[i].item()
                    pred_label = predictions[i].item()
                    hv = hypervectors_signed[i]

                    self.hdc.prototype_accum[true_label] += hv
                    self.hdc.class_counts[true_label] += 1

                    if pred_label >= 0:
                        self.hdc.prototype_accum[pred_label] -= hv
                        self.hdc.class_counts[pred_label] += 1

            self.hdc.finalize_prototypes()

            overall_acc, _, _, _ = self.hdc.evaluate(features_tensor, labels_tensor)
            accuracies.append(overall_acc)
            print(f"Iter {it+1}/{n_iters}, Training Accuracy: {overall_acc*100:.2f}%")
        
        return accuracies

    def predict(self, images: torch.Tensor, similarity_threshold: float = 0.0):
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images.to(self.device))
            predictions, similarities = self.hdc.predict(features, similarity_threshold)
        return predictions, similarities

    def evaluate(self, test_loader: torch.utils.data.DataLoader, similarity_threshold: float = 0.0):
        self.feature_extractor.eval()
        all_features, all_labels, all_predictions, all_similarities = [], [], [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.feature_extractor(images)
                predictions, similarities = self.hdc.predict(features, similarity_threshold)
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())
                all_similarities.append(similarities.cpu())
        
        features_tensor = torch.cat(all_features, dim=0).to(self.device)
        labels_tensor = torch.cat(all_labels, dim=0).to(self.device)
        similarities_tensor = torch.cat(all_similarities, dim=0)
        predictions_tensor = torch.cat(all_predictions, dim=0)
        
        overall_acc, valid_acc, avg_sim, confusion = self.hdc.evaluate(features_tensor, labels_tensor)
        avg_confidence = similarities_tensor.max(dim=1)[0].mean().item()
        
        return {
            'overall_accuracy': overall_acc*100,
            'valid_accuracy': valid_acc*100,
            'average_similarity': avg_sim,
            'average_confidence': avg_confidence,
            'confusion_matrix': confusion.cpu().numpy(),
            'predictions': predictions_tensor.numpy(),
            'similarities': similarities_tensor.numpy()
        }