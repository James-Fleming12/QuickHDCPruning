import copy
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from torch.linalg import norm

class HDCPruner(nn.Module):
    def __init__(self, model):
        """
        model needs the following parameters
        - self.device
        - self.n_classes
        - self.feature_dim
        - self.hd_dim
        - self.hdc.features_to_hypervector(inputs)
        - self.feature_extractor(inputs, return_features=True)
        - self.hdc.hamming_dist(input1, input2)
        """
        super().__init__()
        self.model = model

        self.device = model.device
        self.n_classes = model.n_classes
        self.feature_dim = model.feature_dim
        self.hd_dim = model.hd_dim

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
            class_avgs[valid_mask] = ones_count[valid_mask] / total_count[valid_mask].unsqueeze(1)

        return sims, class_avgs

    def prune_metrics_subset(self, dim: int, data: DataLoader):
        """
        Same as prune_metrics, but breaks up computations as to not hold too much data in memory.
        TODO: finish
        """
        hv_sums = torch.Tensor(dim)
        n = len(data)
        for (inputs, labels) in data:
            hvs = self.model.hdc.features_to_hypervector(inputs)
            hv_sums += hvs.sum(dim=0)
        total = (hv_sums**2 + (n-hv_sums**2)).sum()
        return total / (n**2 * dim)

    def hd_prune(self, data: DataLoader) -> int:
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in data:
                images = images.to(self.device)
                features = self.model.feature_extractor(images, return_features=True)
                
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

        error_term = 0.8

        while high > low:
            valid = True
            mid = low + (high - low) // 2

            diffs, avgs = self.prune_metrics(mid, features_dataloader)

            for i in range(self.n_classes):
                if diffs[i+1] < diffs[0]: # global distance comparisons (maybe include additional noise term?)
                    valid = False

                for j in range(self.n_classes):
                    if i == j: continue
                    if self.expected_hamming(avgs[i], avgs[j]) > diffs[i+1]:
                        valid = False
                    
            if valid:
                high = mid
                if res > mid:
                    res = mid
            else:
                low = mid + 1

        return res
    
    def expected_hamming(self, p, q):
        dim = p.shape[0]
        hamming = (p + q - 2*p*q).sum()
        return 1 - (hamming / dim)

class SimpleMicroHD:
    def __init__(self, model_factory, original_model, target_accuracy=0.85):
        """
        Args:
            model_factory: Function that creates a new model instance given a dimension
                          e.g., lambda dim: HAR_HDC(dataset_type='ucihar', dim=dim)
            original_model: The original trained model to use as reference
            target_accuracy: Minimum acceptable accuracy (0-1)
        """
        self.model_factory = model_factory
        self.original_model = original_model
        self.target_accuracy = target_accuracy
        self.device = original_model.device
        
    def _retrain_model(self, model, data_loader, epochs=5):
        """
        model needs
        - feature_extractor attribute
        - hdc attribute with features_to_hypervector, predict, evaluate methods
        - train_hdc_iterative method OR we use standard iterative training
        """
        if hasattr(model, 'train_hdc_iterative'):
            model.train_hdc_iterative(data_loader, n_iters=epochs)

            all_features, all_labels = [], []
            with torch.no_grad():
                for batch_data, batch_labels in data_loader:
                    batch_data = batch_data.to(self.device)
                    features = model.feature_extractor(batch_data)
                    all_features.append(features.cpu())
                    all_labels.append(batch_labels)
            
            features_tensor = torch.cat(all_features, dim=0).to(self.device)
            labels_tensor = torch.cat(all_labels, dim=0).to(self.device)
            
            overall_acc, _, _, _ = model.hdc.evaluate(features_tensor, labels_tensor)
            return overall_acc
            
        else:
            model.feature_extractor.eval()

            all_features, all_labels = [], []
            with torch.no_grad():
                for batch_data, batch_labels in data_loader:
                    batch_data = batch_data.to(self.device)
                    features = model.feature_extractor(batch_data)
                    all_features.append(features.cpu())
                    all_labels.append(batch_labels)
            
            features_tensor = torch.cat(all_features, dim=0).to(self.device)
            labels_tensor = torch.cat(all_labels, dim=0).to(self.device)

            model.hdc.prototype_accum.zero_()
            model.hdc.class_counts.zero_()
            
            hypervectors = model.hdc.features_to_hypervector(features_tensor)
            hypervectors_signed = hypervectors.int() * 2 - 1

            for i in range(len(labels_tensor)):
                label = labels_tensor[i].item()
                model.hdc.prototype_accum[label] += hypervectors_signed[i]
                model.hdc.class_counts[label] += 1
            model.hdc.finalize_prototypes()

            for epoch in range(epochs):
                predictions, _ = model.hdc.predict(features_tensor)
                misclassified = predictions != labels_tensor
                
                for i in range(len(labels_tensor)):
                    if misclassified[i]:
                        true_label = labels_tensor[i].item()
                        pred_label = predictions[i].item()
                        hv = hypervectors_signed[i]
                        
                        model.hdc.prototype_accum[true_label] += hv
                        model.hdc.class_counts[true_label] += 1
                        
                        if pred_label >= 0:
                            model.hdc.prototype_accum[pred_label] -= hv
                            model.hdc.class_counts[pred_label] += 1
                
                model.hdc.finalize_prototypes()

            overall_acc, _, _, _ = model.hdc.evaluate(features_tensor, labels_tensor)
            return overall_acc
    
    def _copy_feature_extractor(self, source_model, target_model):
        """Copy feature extractor weights from source to target model."""
        if hasattr(source_model, 'feature_extractor') and hasattr(target_model, 'feature_extractor'):
            target_model.feature_extractor.load_state_dict(
                copy.deepcopy(source_model.feature_extractor.state_dict())
            )
        
        for attr_name in ['encoder', 'classifier', 'projector']:
            if hasattr(source_model, attr_name) and hasattr(target_model, attr_name):
                target_attr = getattr(target_model, attr_name)
                source_attr = getattr(source_model, attr_name)
                target_attr.load_state_dict(copy.deepcopy(source_attr.state_dict()))
    
    def _get_dimension_space(self, original_dim):
        """Generate binary search-friendly dimension space."""
        space = []
        
        min_exp = 6
        max_exp = 14
        
        for exp in range(min_exp, max_exp + 1):
            dim = 2 ** exp
            space.append(dim)

        if original_dim not in space:
            space.append(original_dim)
        
        space.sort()
        return space
    
    def hd_tune(self, data_loader, retrain_epochs=5):
        """
        Binary search for optimal dimension.
        """
        if hasattr(self.original_model, 'hd_dim'):
            original_dim = self.original_model.hd_dim
        elif hasattr(self.original_model, 'hdc') and hasattr(self.original_model.hdc, 'dimension'):
            original_dim = self.original_model.hdc.dimension
        else:
            raise ValueError("Cannot determine original dimension from model")

        dim_space = self._get_dimension_space(original_dim)

        low = 0
        high = len(dim_space) - 1
        best_dim = original_dim

        results_cache = {}
        
        while low <= high:
            mid = (low + high) // 2
            test_dim = dim_space[mid]
            
            if test_dim not in results_cache:
                test_model = self.model_factory(test_dim)
                test_model.to(self.device)
                
                self._copy_feature_extractor(self.original_model, test_model)

                accuracy = self._retrain_model(test_model, data_loader, retrain_epochs)
                results_cache[test_dim] = accuracy

                del test_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                accuracy = results_cache[test_dim]
            
            if accuracy >= self.target_accuracy and test_dim < best_dim:
                best_dim = test_dim

            if accuracy >= self.target_accuracy:
                high = mid - 1
            else:
                low = mid + 1
        
        return best_dim

def create_microhd(model_class, original_model, **constructor_kwargs):
    def model_factory(dim):
        model = model_class(dim=dim, **constructor_kwargs)

        if hasattr(original_model, 'feature_extractor') and hasattr(model, 'feature_extractor'):
            model.feature_extractor.load_state_dict(
                original_model.feature_extractor.state_dict()
            )
        
        return model
    
    return SimpleMicroHD(
        model_factory=model_factory,
        original_model=original_model
    )



class HDCClassifier(nn.Module):
    def __init__(self, n_classes: int, dimension: int = 5000, similarity_threshold: float = 0.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dimension
        self.n_classes = n_classes
        
        self.register_buffer('prototypes', torch.zeros(n_classes, dimension, dtype=torch.bool, device=self.device))
        self.register_buffer('prototype_accum', torch.zeros(n_classes, dimension, dtype=torch.int32, device=self.device))
        self.register_buffer('class_counts', torch.zeros(n_classes, dtype=torch.long, device=self.device))

        self.similarity_thresh = similarity_threshold

        self.feature_basis = None
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
                self.prototypes[c] = self.prototype_accum[c] > 0

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

def main():
    pass

if __name__=="__main__":
    main()