import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from image_model import CNNExtractor, HDCImageClassifier

class HDCPruner(nn.Module):
    def __init__(self, model: HDCImageClassifier):
        super().__init__()
        self.model = model

        self.device = model.device
        self.n_classes = model.n_classes
        self.feature_dim = model.feature_dim
        self.hd_dim = model.hd_dim

    def forward(self, x) -> int:
        pass

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
                features = self.model.feature_extractor(images)
                
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

            for i in range(self.n_classes):
                if diffs[i+1] < diffs[0]: # global distance comparisons (maybe include additional noise term?)
                    valid = False

                for j in range(self.n_classes):
                    if i == j: continue
                    if self.model.hdc.hamming_dist(avgs[i].to(torch.bool), avgs[j].to(torch.bool)) > diffs[i+1]:
                        valid = False

            if valid:
                high = mid
                if res > mid:
                    res = mid
            else:
                low = mid + 1

        return res

def main():
    pass

if __name__=="__main__":
    main()