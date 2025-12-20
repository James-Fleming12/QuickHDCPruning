from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class HDCClassifier(nn.Module):
    def __init__(self, n_classes: int, dimension: int = 5000, similarity_threshold: float = 0.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dimension
        self.n_classes = n_classes
        
        self.prototypes = torch.zeros(n_classes, dimension, dtype=torch.bool, device=self.device)
        self.prototype_accum = torch.zeros(n_classes, self.dim, dtype=torch.int32, device=self.device)
        self.class_counts = torch.zeros(n_classes, device=self.device)

        self.feature_basis = None

        self.similarity_thresh = similarity_threshold

        self.random_projection = None

    def initialize_basis_vectors(self, feature_dim:int) -> None:
        self.feature_basis = torch.rand((feature_dim, self.dim), device=self.device) < 0.5

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a ^ b

    def bundle(self, x: torch.Tensor) -> torch.Tensor:
        sum_bits = x.sum(dim=0, dtype=torch.float32)
        threshold = x.shape[0] / 2
        return sum_bits > threshold

    def hamming_dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        differences = a^b
        hamming_dist = differences.sum(dim=-1, dtype=torch.float32)

        similarity = 1.0 - (hamming_dist / self.dim)
        return similarity
    
    def initialize_projection(self, feature_dim: int):
        self.random_projection = torch.randint(0, 2, (feature_dim, self.dim), device=self.device).float() * 2 - 1

    def features_to_hypervector(self, features: torch.Tensor) -> torch.Tensor:
        _, feature_dim = features.shape
        if self.random_projection is None:
            self.initialize_projection(feature_dim)

        projected = features @ self.random_projection

        hvs = projected > 0
        return hvs.to(torch.bool)

    def train(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        hypervectors = self.features_to_hypervector(features)
        for i in range(len(labels)):
            label = labels[i].item()
            hv = hypervectors[i]
            hv_signed = hv.float() * 2 - 1

            self.prototype_accum[label] += hv_signed
            self.class_counts[label] += 1

    def finalize_prototypes(self):
        for c in range(self.n_classes):
            if self.class_counts[c] > 0:
                self.prototypes[c] = self.prototype_accum[c] >= 0

    def predict(self, features: torch.Tensor, similarity_thresh: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        hypervectors = self.features_to_hypervector(features)
        batch_size = hypervectors.shape[0]
        
        similarities = torch.zeros(batch_size, self.n_classes, device=self.device)
        
        for c in range(self.n_classes):
            if self.class_counts[c] > 0:
                proto_expanded = self.prototypes[c].unsqueeze(0).expand(batch_size, -1)
                similarities[:, c] = self.hamming_dist(hypervectors, proto_expanded)
        
        max_sims, preds = torch.max(similarities, dim=1)
        if similarity_thresh > 0:
            preds[max_sims < similarity_thresh] = -1
        
        return preds, similarities
    
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        predictions, similarities = self.predict(features, self.similarity_thresh)
        
        valid_mask = predictions != -1
        if valid_mask.any():
            valid_accuracy = (predictions[valid_mask] == labels[valid_mask]).float().mean().item()
        else:
            valid_accuracy = 0.0
        
        overall_accuracy = (predictions == labels).float().mean().item()
        
        correct_mask = predictions == labels
        if correct_mask.any():
            avg_similarity = similarities[correct_mask].max(dim=1)[0].mean().item()
        else:
            avg_similarity = 0.0
        
        confusion = torch.zeros(self.n_classes, self.n_classes, device=self.device)
        for true, pred in zip(labels.cpu(), predictions.cpu()):
            if pred >= 0:
                confusion[true, pred] += 1
        
        return overall_accuracy, valid_accuracy, avg_similarity, confusion

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

    def forward(self, x):
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

    def train_hdc(self, train_loader: torch.utils.data.DataLoader, n_epochs: int = 1):
        accuracies = []
        for epoch in range(n_epochs):
            self.feature_extractor.eval()
            all_features, all_labels = [], []
            
            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    features = self.feature_extractor(images)
                    all_features.append(features.cpu())
                    all_labels.append(labels)
            
            features_tensor = torch.cat(all_features, dim=0).to(self.device)
            labels_tensor = torch.cat(all_labels, dim=0).to(self.device)
            
            self.hdc.train(features_tensor, labels_tensor)
            self.hdc.finalize_prototypes()
            
            overall_acc, _, _, _ = self.hdc.evaluate(features_tensor, labels_tensor)
            accuracies.append(overall_acc)
            print(f"HDC epoch {epoch+1}/{n_epochs}, Training Accuracy: {overall_acc*100:.2f}%")
        
        return accuracies
    
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
    
    def prune_metrics(self, dim: int, data: DataLoader):
        """
        returns diffs, svgs
        diffs[0] is average global distance, diffs[1-n_classes+1] are average label distances per class
        avgs[0-n_classes] are average hypervector per class
        """
        device = self.device
        
        random_projection = (torch.randint(0, 2, (self.feature_dim, dim), device=device) * 2 - 1).float()

        hv_sums = torch.zeros(dim, device=device)
        ones_count = torch.zeros(self.n_classes, dim, device=device)
        total_count = torch.zeros(self.n_classes, device=device)
        n = 0

        for features, labels in data:
            features, labels = features.to(device), labels.to(device)

            projected = features @ random_projection
            hvs = (projected > 0).float()
            batch_size = hvs.shape[0]

            hv_sums += hvs.sum(dim=0)
            ones_count.index_add_(0, labels, hvs)
            total_count.index_add_(0, labels, torch.ones(batch_size, device=device))
            n += batch_size

        hv_sums_sq = hv_sums.square()
        total = (hv_sums_sq + (n - hv_sums).square()).sum()
        avg_sim = total / (n**2 * dim)
        avg_sim_tensor = torch.tensor([avg_sim], device=device)

        valid_mask = total_count > 0
        valid_indices = torch.where(valid_mask)[0]

        class_sims = torch.zeros(self.n_classes, device=device)
        
        if len(valid_indices) > 0:
            valid_counts = total_count[valid_indices]
            valid_sums = ones_count[valid_indices]

            sums_sq = valid_sums.square()
            diff = valid_counts.unsqueeze(1) - valid_sums
            class_totals = (sums_sq + diff.square()).sum(dim=1)
            valid_sims = class_totals / (valid_counts.square() * dim)

            class_sims[valid_indices] = valid_sims

        sims = torch.cat([avg_sim_tensor, class_sims])

        class_avgs = torch.zeros(self.n_classes, dim, device=device)
        if valid_mask.any():
            class_avgs[valid_mask] = (ones_count[valid_mask] > (total_count[valid_mask].unsqueeze(1) / 2)).float()
        
        return sims, class_avgs

    def prune_metrics_subset(self, dim: int, data: DataLoader):
        """
        Same as prune_metrics, but breaks up computations as to not hold too much data in memory.
        TODO: finish
        """
        hv_sums = torch.Tensor(dim)
        n = len(data)
        for (inputs, labels) in data:
            hvs = self.hdc.features_to_hypervector(inputs)
            hv_sums += hvs.sum(dim=0)
        total = (hv_sums**2 + (n-hv_sums**2)).sum()
        return total / (n**2 * dim)

    def hd_prune(self, data: DataLoader) -> int:
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in data:
                images = images.to(self.device)
                features = self.feature_extractor(images)
                
                all_features.append(features)
                all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        features_dataset = torch.utils.data.TensorDataset(all_features, all_labels)

        features_dataloader = torch.utils.data.DataLoader(features_dataset, batch_size=data.batch_size, num_workers=0)

        ref_diffs, ref_avgs = self.prune_metrics(self.hd_dim, features_dataloader)

        high = self.hd_dim
        low = 1
        res = self.hd_dim
        while high > low:
            valid = True
            mid = low + (high - low) // 2

            diffs, avgs = self.prune_metrics(mid, features_dataloader)
            print(diffs)

            for i in range(self.n_classes):
                if diffs[i+1] < diffs[0]: # global distance comparisons (maybe include additional noise term?)
                    valid = False

                for j in range(self.n_classes):
                    if i == j: continue
                    if self.hdc.hamming_dist(avgs[i].to(torch.bool), avgs[j].to(torch.bool)) > diffs[i+1]:
                        valid = False

            print(f"Dimesion {mid} checked: {"Valid" if valid else "Invalid"}")
            
            if valid:
                high = mid
                if res > mid:
                    res = mid
            else:
                low = mid + 1

        return res
