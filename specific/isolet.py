import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from main import HDCClassifier, HDCPruner, SimpleMicroHD, create_microhd

class ISOLETDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        """
        Args:
            data_path (str): Path to directory containing ISOLET files
            train (bool): If True, load training set, else test set
            transform (callable, optional): Optional transform to apply
        """
        self.transform = transform
        self.train = train

        train_files = ['isolet1+2+3+4.data']
        test_files = ['isolet5.data']

        files_to_load = train_files if train else test_files
        
        data_list = []
        labels_list = []
        
        for filename in files_to_load:
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                data = pd.read_csv(filepath, header=None, sep=r',\s+', engine='python')

                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
                
                data_list.append(X)
                labels_list.append(y)
            else:
                print(f"Warning: {filepath} not found")
        
        if data_list:
            self.data = np.vstack(data_list).astype(np.float32)
            self.labels = np.hstack(labels_list).astype(np.int64)
        else:
            raise FileNotFoundError(f"No ISOLET data files found in {data_path}")

        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.std[self.std == 0] = 1
        self.data = (self.data - self.mean) / self.std

        self.data = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.labels - 1)  # Convert to 0-25 range
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
    
    def get_feature_dim(self):
        return self.data.shape[1]
    
    def get_num_classes(self):
        return len(torch.unique(self.labels))
    
class ISOLETFeatureExtractor(nn.Module):
    def __init__(self, input_dim=617, feature_dim=128, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Input feature dimension (617 for ISOLET)
            feature_dim (int): Dimension of extracted features
            dropout_rate (float): Dropout rate for regularization
        """
        super(ISOLETFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(feature_dim, 26)
        
    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        
        if return_features:
            return features
        return logits
    
    def extract_features(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return features

class ISOLET_HDC(nn.Module):
    def __init__(self, dim=5000):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_classes = 26
        self.hd_dim = dim
        self.feature_dim = 128

        self.feature_extractor = ISOLETFeatureExtractor(feature_dim=self.feature_dim)
        self.hdc = HDCClassifier(self.n_classes, dimension=self.hd_dim)
        self.feature_extractor.to(self.device)
        self.hdc.to(self.device)

    def init_feature_extractor(self, model_path: str):
        self.feature_extractor.load_state_dict(torch.load(model_path, map_location=self.device))
        self.feature_extractor.eval()
        print(f"Loaded Feature Extractor from {model_path}")
    
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

def isolet_config():
    return {
        "data_path": "data/isolet",
        "model_save_path": "models/isolet_feature_extractor.pth",
        "batch_size": 16,
        "hdc_save_path": "models/isolet_hdc.pth",
        "new_save_path": "models/isolet_pruned.pth",
    }

def train_isolet_extractor():
    config = isolet_config()
    data_path = config['data_path']
    model_save_path = config['model_save_path']
    batch_size = config['batch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr = 0.001
    weight_decay = 1e-4
    epochs = 50

    try:
        train_dataset = ISOLETDataset(data_path, train=True)
        val_dataset = ISOLETDataset(data_path, train=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    input_dim = train_dataset.get_feature_dim()
    model: ISOLETFeatureExtractor = ISOLETFeatureExtractor(input_dim=input_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    model.train()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        for index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            logits = model.forward(data, return_features=False)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: {train_loss / num_batches}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data, return_features=False)
                loss = criterion(logits, target)
                
                val_loss += loss.item() * data.size(0)
                pred = logits.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += data.size(0)
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Best model saved with val loss {val_loss:.4f}')

def train_isolet_hdc(dim: int, pruned: bool = False, random_projection=None) -> ISOLET_HDC:
    config = isolet_config()
    data_path = config['data_path']
    model_save_path = config['model_save_path']
    batch_size = config['batch_size']
    hdc_save_path = config['hdc_save_path']
    new_save_path = config['new_save_path']

    try:
        train_dataset = ISOLETDataset(data_path, train=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    hdc = ISOLET_HDC(dim=dim)
    hdc.init_feature_extractor(model_save_path)

    if random_projection is not None:
        hdc.hdc.random_projection = random_projection

    accs = hdc.train_hdc_iterative(train_loader)

    if pruned:
        print(f"Model saved with accuracy {accs} in {new_save_path}")
        torch.save(hdc.state_dict(), new_save_path)
    else:
        print(f"Model saved with accuracy {accs} in {hdc_save_path}")
        torch.save(hdc.state_dict(), hdc_save_path)

    return hdc

def eval_isolet(hdc: ISOLET_HDC):
    config = isolet_config()
    data_path = config['data_path']
    model_save_path = config['model_save_path']
    batch_size = config['batch_size']

    try:
        val_dataset = ISOLETDataset(data_path, train=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

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

def prune_isolet():
    config = isolet_config()
    data_path = config['data_path']
    model_save_path = config['hdc_save_path']
    batch_size = config['batch_size']

    try:
        train_dataset = ISOLETDataset(data_path, train=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    hdc = ISOLET_HDC()
    hdc.load_state_dict(torch.load(model_save_path))

    old_dim = hdc.hd_dim

    pruner = HDCPruner(hdc)
    new_dim, proj = pruner.hd_prune(train_loader)
    print(f"Achieved new dimension {new_dim} from original {old_dim}")

    return new_dim, proj

def micro_isolet():
    config = isolet_config()
    data_path = config['data_path']
    model_save_path = config['hdc_save_path']
    batch_size = config['batch_size']

    try:
        train_dataset = ISOLETDataset(data_path, train=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    hdc = ISOLET_HDC()
    hdc.load_state_dict(torch.load(model_save_path))

    old_dim = hdc.hd_dim

    tuner = create_microhd(ISOLET_HDC, hdc)
    new_dim = tuner.hd_tune(train_loader)

    print(f"MicroHD achieved new dimension {new_dim} from original {old_dim}")

    return new_dim

def main():
    pass

if __name__=="__main__":
    main()