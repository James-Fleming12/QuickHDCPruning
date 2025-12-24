import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from scipy import stats, fft
from typing import List, Tuple, Optional

from main import HDCClassifier, HDCPruner

class HARFeatureExtractor:
    """
    Feature extractor for UCIHAR and PAMAP datasets.
    Returns one feature vector per window for classification.
    """
    def __init__(self, sampling_rate: float = 50.0, window_size: float = 2.5, overlap: float = 0.5):
        self.sampling_rate = sampling_rate
        self.window_size = int(window_size * sampling_rate)
        self.overlap = overlap
        self.step_size = int(self.window_size * (1 - overlap))
        self.feature_names_ = None
        
    def segment_and_extract(self, data: np.ndarray, labels: Optional[np.ndarray] = None, subject_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment data into windows and extract features.
        Returns X (features) and y (labels) for classification.
        """
        windows = []
        window_labels = []
        
        n_samples = len(data)

        for start in range(0, n_samples - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window = data[start:end]
            windows.append(window)
            
            if labels is not None:
                window_label = stats.mode(labels[start:end], keepdims=False)[0]
                window_labels.append(window_label)
        
        features_list = []
        for window in windows:
            features = self._extract_window_features(window)
            features_list.append(features)
        
        X = np.array(features_list)
        y = np.array(window_labels) if window_labels else None
        
        if self.feature_names_ is None:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        return X, y
    
    def _extract_window_features(self, window: np.ndarray) -> np.ndarray:
        """Extract features from a single window."""
        features = []
        
        for col in range(window.shape[1]):
            signal = window[:, col]

            features.extend([
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.ptp(signal),
                np.median(signal),
                np.mean(np.abs(signal - np.mean(signal))),
                np.sum(signal ** 2) / len(signal),
                np.sqrt(np.mean(signal ** 2)),
                stats.skew(signal),
                stats.kurtosis(signal),
                np.sum(np.diff(np.sign(signal)) != 0) / len(signal), 
            ])
            
            features.extend(np.percentile(signal, [25, 50, 75, 90]))
            
            n = len(signal)
            if n > 1:
                fft_vals = fft.fft(signal)
                magnitudes = np.abs(fft_vals[:n//2])
                if len(magnitudes) > 0:
                    features.extend([
                        np.mean(magnitudes),
                        np.std(magnitudes),
                        np.sum(magnitudes ** 2) / len(magnitudes),
                    ])
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])

        if window.shape[1] > 1:
            corr_matrix = np.corrcoef(window.T)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            features.extend(upper_tri)
        
        return np.array(features)

class UCIHARDataset(Dataset):
    def __init__(self, data_path: str, train: bool = True, window_size: float = 2.5, overlap: float = 0.5, transform=None):
        self.transform = transform
        self.train = train
        
        self.extractor = HARFeatureExtractor(
            sampling_rate=50.0,
            window_size=window_size,
            overlap=overlap
        )
        
        X, y = self._load_ucihar_data(data_path, train)
        
        self.features, self.labels = self.extractor.segment_and_extract(X, y)
        
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.std[self.std == 0] = 1
        self.features = (self.features - self.mean) / self.std
        
        self.features = torch.from_numpy(self.features.astype(np.float32))
        self.labels = torch.from_numpy(self.labels.astype(np.int64) - 1)
        
    def _load_ucihar_data(self, data_path: str, train: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Load UCIHAR dataset."""
        if train:
            x_file = os.path.join(data_path, 'train', 'X_train.txt')
            y_file = os.path.join(data_path, 'train', 'y_train.txt')
        else:
            x_file = os.path.join(data_path, 'test', 'X_test.txt')
            y_file = os.path.join(data_path, 'test', 'y_test.txt')
        
        X = np.loadtxt(x_file)
        y = np.loadtxt(y_file).astype(int)
        
        return X, y
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, label
    
    def get_feature_dim(self):
        return self.features.shape[1]
    
    def get_num_classes(self):
        return len(torch.unique(self.labels))


class PAMAPDataset(Dataset):
    def __init__(self, data_path: str, train: bool = True, window_size: float = 2.5, overlap: float = 0.5, transform=None, selected_columns: Optional[List[int]] = None):
        self.transform = transform
        
        self.extractor = HARFeatureExtractor(
            sampling_rate=100.0,
            window_size=window_size,
            overlap=overlap
        )
        
        X, y, subjects = self._load_pamap_data(data_path, selected_columns)
        
        if train:
            train_mask = np.isin(subjects, [1, 2, 3, 4, 5, 6])
        else:
            train_mask = np.isin(subjects, [7, 8, 9])
        
        X = X[train_mask]
        y = y[train_mask]

        self.features, self.labels = self.extractor.segment_and_extract(X, y)

        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.std[self.std == 0] = 1
        self.features = (self.features - self.mean) / self.std

        self.features = torch.from_numpy(self.features.astype(np.float32))
        self.labels = torch.from_numpy(self.labels.astype(np.int64))

        unique_labels = np.unique(self.labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = torch.tensor([label_map[label.item()] for label in self.labels])
        
    def _load_pamap_data(self, data_path: str, selected_columns: Optional[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load PAMAP dataset."""
        data = pd.read_csv(data_path, sep=' ', header=None)

        if selected_columns is None:
            selected_columns = list(range(3, 15))

        X = data.iloc[:, selected_columns].values
        y = data.iloc[:, 1].values
        subjects = data.iloc[:, 0].values
        
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        subjects = subjects[valid_mask]

        known_mask = y > 0
        X = X[known_mask]
        y = y[known_mask]
        subjects = subjects[known_mask]
        
        return X, y, subjects
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
            
        return features, label
    
    def get_feature_dim(self):
        return self.features.shape[1]
    
    def get_num_classes(self):
        return len(torch.unique(self.labels))

class HARFeatureExtractorNN(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int = 128, dropout_rate: float = 0.3):
        super(HARFeatureExtractorNN, self).__init__()
        
        self.feature_dim = feature_dim

        hidden1 = min(512, max(256, input_dim // 2))
        hidden2 = min(256, max(128, hidden1 // 2))
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    def extract_features(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return features
    
class HAR_HDC(nn.Module):
    def __init__(self, dataset_type: str = 'ucihar',
                 dim: int = 5000,
                 feature_dim: int = 128):
        """dataset_type = 'ucihar' or 'pamap'"""
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_type = dataset_type

        if dataset_type == 'ucihar':
            self.n_classes = 6  # UCIHAR has 6 activities
        elif dataset_type == 'pamap':
            self.n_classes = 12  # PAMAP has 12 main activities
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        self.hd_dim = dim
        self.feature_dim = feature_dim

        self.feature_extractor = None
        self.hdc = HDCClassifier(self.n_classes, dimension=self.hd_dim)
        
        self.to(self.device)

    def init_feature_extractor(self, input_dim: int, model_path: Optional[str] = None):
        self.feature_extractor = HARFeatureExtractorNN(
            input_dim=input_dim,
            feature_dim=self.feature_dim
        ).to(self.device)
        
        if model_path is not None and os.path.exists(model_path):
            self.feature_extractor.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.feature_extractor.eval()
            print(f"Loaded Feature Extractor from {model_path}")

    def train_hdc_iterative(self, train_loader: DataLoader, n_iters: int = 3):
        """Train HDC classifier iteratively."""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized. Call init_feature_extractor() first.")
        
        self.feature_extractor.eval()
        
        all_features, all_labels = [], []
        with torch.no_grad():
            for data, labels in train_loader:
                data = data.to(self.device)
                features = self.feature_extractor(data)
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
    
    def predict(self, data: torch.Tensor, similarity_threshold: float = 0.0):
        """Make predictions."""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized.")
        
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(data.to(self.device))
            predictions, similarities = self.hdc.predict(features, similarity_threshold)
        return predictions, similarities
    
    def evaluate(self, test_loader: DataLoader, similarity_threshold: float = 0.0):
        """Evaluate model."""
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized.")
        
        self.feature_extractor.eval()
        all_features, all_labels, all_predictions, all_similarities = [], [], [], []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                features = self.feature_extractor(data)
                predictions, similarities = self.hdc.predict(features, similarity_threshold)
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())
                all_similarities.append(similarities.cpu())
        
        features_tensor = torch.cat(all_features, dim=0).to(self.device)
        labels_tensor = torch.cat(all_labels, dim=0).to(self.device)
        
        overall_acc, valid_acc, avg_sim, confusion = self.hdc.evaluate(features_tensor, labels_tensor)
        
        return {
            'overall_accuracy': overall_acc * 100,
            'valid_accuracy': valid_acc * 100,
            'average_similarity': avg_sim,
            'confusion_matrix': confusion.cpu().numpy()
        }

def har_config(dataset_type: str = 'ucihar'):
    """Get configuration for HAR datasets."""
    if dataset_type == 'ucihar':
        return {
            "data_path": "data/ucihar",
            "model_save_path": "models/ucihar_feature_extractor.pth",
            "hdc_save_path": "models/ucihar_hdc.pth",
            "batch_size": 32,
            "window_size": 2.5,
            "overlap": 0.5
        }
    elif dataset_type == 'pamap':
        return {
            "data_path": "data/pamap/Protocol.dat",
            "model_save_path": "models/pamap_feature_extractor.pth",
            "hdc_save_path": "models/pamap_hdc.pth",
            "batch_size": 32,
            "window_size": 2.5,
            "overlap": 0.5
        }

def train_har_extractor(dataset_type: str = 'ucihar'):
    """Train neural network feature extractor for HAR."""
    config = har_config(dataset_type)
    data_path = config['data_path']
    model_save_path = config['model_save_path']
    batch_size = config['batch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset_type == 'ucihar':
        train_dataset = UCIHARDataset(data_path, train=True)
        val_dataset = UCIHARDataset(data_path, train=False)
    elif dataset_type == 'pamap':
        train_dataset = PAMAPDataset(data_path, train=True)
        val_dataset = PAMAPDataset(data_path, train=False)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = train_dataset.get_feature_dim()
    model = HARFeatureExtractorNN(input_dim=input_dim).to(device)

    classifier = nn.Linear(128, train_dataset.get_num_classes()).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), 
                           lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        classifier.train()
        
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            features = model(data)
            logits = classifier(features)
            loss = criterion(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        model.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                features = model(data)
                logits = classifier(features)
                loss = criterion(logits, target)
                
                val_loss += loss.item() * data.size(0)
                pred = logits.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
        
        val_loss /= len(val_dataset)
        val_acc = 100. * val_correct / len(val_dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Epoch {epoch+1}: Saved model with val loss {val_loss:.4f}, acc {val_acc:.2f}%')

def train_har_hdc(dataset_type: str = 'ucihar', dim: int = 5000):
    """Train HDC model for HAR."""
    config = har_config(dataset_type)
    data_path = config['data_path']
    model_save_path = config['model_save_path']
    hdc_save_path = config['hdc_save_path']
    batch_size = config['batch_size']

    if dataset_type == 'ucihar':
        train_dataset = UCIHARDataset(data_path, train=True)
    elif dataset_type == 'pamap':
        train_dataset = PAMAPDataset(data_path, train=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    hdc_model = HAR_HDC(dataset_type=dataset_type, dim=dim)
    hdc_model.init_feature_extractor(
        input_dim=train_dataset.get_feature_dim(),
        model_path=model_save_path
    )

    accuracies = hdc_model.train_hdc_iterative(train_loader, n_iters=3)
    
    torch.save(hdc_model.state_dict(), hdc_save_path)
    print(f"HDC model saved with accuracies: {[acc*100 for acc in accuracies]}")
    
    return hdc_model

def prune_har(dataset_type: str = 'ucihar'):
    config = har_config(dataset_type)()
    data_path = config['data_path']
    model_save_path = config['hdc_save_path']
    batch_size = config['batch_size']

    if dataset_type == 'ucihar':
        train_dataset = UCIHARDataset(data_path, train=True)
    elif dataset_type == 'pamap':
        train_dataset = PAMAPDataset(data_path, train=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    hdc = HAR_HDC(dataset_type=dataset_type)
    hdc.load_state_dict(torch.load(model_save_path))

    old_dim = hdc.hd_dim

    pruner = HDCPruner(hdc)
    new_dim = pruner.hd_prune(train_loader)
    print(f"Achieved new dimension {new_dim} from original {old_dim}")

    return new_dim

def eval_har(hdc_model: HAR_HDC, dataset_type: str = 'ucihar'):
    """Evaluate HDC model."""
    config = har_config(dataset_type)
    data_path = config['data_path']
    batch_size = config['batch_size']
    
    if dataset_type == 'ucihar':
        test_dataset = UCIHARDataset(data_path, train=False)
    elif dataset_type == 'pamap':
        test_dataset = PAMAPDataset(data_path, train=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    results = hdc_model.evaluate(test_loader)
    
    print("\n" + "="*50)
    print(f"{dataset_type.upper()} EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Valid Accuracy: {results['valid_accuracy']:.2f}%")
    print(f"Average Similarity: {results['average_similarity']:.4f}")

def main():
    pass

if __name__=="__main__":
    main()