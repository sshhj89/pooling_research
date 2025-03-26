import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinkhorn Algorithm for Optimal Transport with delta_u check
def sinkhorn(K, mu_s, mu_t, num_iters=10, epsilon=1e-6, delta_thresh=1e-3):
    u = torch.ones_like(mu_s)  # Shape: (B, HW)
    v = torch.ones_like(mu_t)  # Shape: (B, num_proxies)
    for _ in range(num_iters):
        u_prev = u.clone()
        u = mu_s / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, HW)
        v = mu_t / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, num_proxies)
        if torch.max(torch.abs(u - u_prev)) < delta_thresh:
            break
    T = torch.bmm(torch.bmm(torch.diag_embed(u), K), torch.diag_embed(v))  # Shape: (B, HW, num_proxies)
    return T

# from here https://michielstock.github.io/OptimalTransport/
def compute_optimal_transport(K, mu_s, mu_t, num_iters=50, epsilon=1e-6):
    v = torch.ones_like(mu_t)  # Shape: (B, num_proxies)
    u = torch.zeros_like(mu_s)  # Shape: (B, HW)
    # K = (B, HW, num_proxies)
    for _ in range(num_iters):
        u_prev = u.clone()
        u = mu_s / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, HW)
        v = mu_t / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, num_proxies)
        if torch.max(torch.abs(u - u_prev)) < epsilon:
            break

    T = torch.bmm(torch.bmm(torch.diag_embed(u), K), torch.diag_embed(v))  # Shape: (B, HW, num_proxies)
    return T


# Multi-Proxy Wasserstein Classifier class
class MultiProxyWassersteinClassifier(torch.nn.Module):
    def __init__(self, feature_dim, num_classes, num_proxies):
        super().__init__()
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, num_proxies, feature_dim))  # Shape: (num_classes, num_proxies, feature_dim)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, features):
        B, C, H, W = features.shape  # Batch, Channels, Height, Width
        HW = H * W
        features = features.view(B, C, HW).permute(0, 2, 1)  # Shape: (B, HW, C)
        scores = []

        mu_s = torch.full((B, HW), 1.0 / HW).to(features.device)  # Uniform distribution over spatial features, shape: (B, HW)
        mu_t = torch.full((B, self.proxies.shape[1]), 1.0 / self.proxies.shape[1]).to(features.device)  # Uniform distribution over proxies, shape: (B, num_proxies)

        for cls in range(self.proxies.shape[0]):
            proxies = self.proxies[cls].unsqueeze(0).expand(B, -1, -1)  # Shape: (B, num_proxies, C)
            sim = self.cosine_similarity(features.unsqueeze(2), proxies.unsqueeze(1))  # Shape: (B, HW, num_proxies)

            cost_matrix = 1 - sim  # Shape: (B, HW, num_proxies)
            K = torch.exp(-cost_matrix / 0.1)  # Shape: (B, HW, num_proxies)

            T = compute_optimal_transport(K, mu_s, mu_t)  # Shape: (B, HW, num_proxies)
            classification_score = (T * sim).sum(dim=(1, 2))  # Aggregate scores, shape: (B,)
            scores.append(classification_score)

        logits = torch.stack(scores, dim=1)  # Shape: (B, num_classes)
        return logits

# Example Usage
if __name__ == "__main__":
    feature_dim = 64
    num_classes = 10
    num_proxies = 4

    classifier = MultiProxyWassersteinClassifier(feature_dim, num_classes, num_proxies)
    dummy_features = torch.randn(8, feature_dim, 8, 8)  # Example batch size 8, feature map 8x8

    logits = classifier(dummy_features)
    print(logits.shape)  # Output: torch.Size([8, 10])