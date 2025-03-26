import torch
import torch.nn.functional as F
#
# Sinkhorn Algorithm for Optimal Transport with delta_u check
# def sinkhorn(K, mu_s, mu_t, num_iters=10, epsilon=1e-6, delta_thresh=1e-3):
#     u = torch.ones_like(mu_s)  # Shape: (B, HW)
#     v = torch.ones_like(mu_t)  # Shape: (B, num_proxies)
#     for _ in range(num_iters):
#         u_prev = u.clone()
#         u = mu_s / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, HW)
#         v = mu_t / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, num_proxies)
#         if torch.max(torch.abs(u - u_prev)) < delta_thresh:
#             break
#     T = torch.bmm(torch.bmm(torch.diag_embed(u), K), torch.diag_embed(v))  # Shape: (B, HW, num_proxies)
#     return T

# # Multi-Proxy Wasserstein Classifier class
# class MultiProxyWassersteinClassifier(torch.nn.Module):
#     def __init__(self, feature_dim, num_classes, num_proxies):
#         super().__init__()
#         self.proxies = torch.nn.Parameter(torch.randn(num_classes, num_proxies, feature_dim))  # Shape: (num_classes, num_proxies, feature_dim)
#         self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
#
#     def forward(self, features):
#         B, C, H, W = features.shape  # Batch, Channels, Height, Width
#         HW = H * W
#         features = features.view(B, C, HW).permute(0, 2, 1)  # Shape: (B, HW, C)
#         scores = []
#
#         mu_s = torch.full((B, HW), 1.0 / HW).to(features.device)  # Uniform distribution over spatial features, shape: (B, HW)
#         mu_t = torch.full((B, self.proxies.shape[1]), 1.0 / self.proxies.shape[1]).to(features.device)  # Uniform distribution over proxies, shape: (B, num_proxies)
#
#         for cls in range(self.proxies.shape[0]):
#             proxies = self.proxies[cls].unsqueeze(0).expand(B, -1, -1)  # Shape: (B, num_proxies, C)
#             sim = self.cosine_similarity(features.unsqueeze(2), proxies.unsqueeze(1))  # Shape: (B, HW, num_proxies)
#
#             cost_matrix = 1 - sim  # Shape: (B, HW, num_proxies)
#             K = torch.exp(-cost_matrix / 0.1)  # Shape: (B, HW, num_proxies)
#
#             T = sinkhorn(K, mu_s, mu_t)  # Shape: (B, HW, num_proxies)
#             classification_score = (T * sim).sum(dim=(1, 2))  # Aggregate scores, shape: (B,)
#             scores.append(classification_score)
#
#         logits = torch.stack(scores, dim=1)  # Shape: (B, num_classes)
#         return logits
#
# # Example Usage
# if __name__ == "__main__":
#     feature_dim = 64
#     num_classes = 10
#     num_proxies = 4
#
#     classifier = MultiProxyWassersteinClassifier(feature_dim, num_classes, num_proxies)
#     dummy_features = torch.randn(8, feature_dim, 8, 8)  # Example batch size 8, feature map 8x8
#
#     logits = classifier(dummy_features)
#     print(logits.shape)  # Output: torch.Size([8, 10])

#
# def test_sinkhorn():
#     B, HW, num_proxies = 2, 4, 3
#
#     K = torch.rand(B, HW, num_proxies).clamp(min=1e-3)
#     mu_s = torch.full((B, HW), 1.0 / HW)
#     mu_t = torch.full((B, num_proxies), 1.0 / num_proxies)
#
#     T = sinkhorn(K, mu_s, mu_t, num_iters=50)
#
#     # Check if T is a valid transportation matrix (rows and columns sum close to mu_s and mu_t)
#     row_sums = T.sum(dim=-1)
#     col_sums = T.sum(dim=-2)
#
#     assert torch.allclose(row_sums, mu_s, atol=1e-3), f"Row sums mismatch: {row_sums} vs {mu_s}"
#     assert torch.allclose(col_sums, mu_t, atol=1e-3), f"Column sums mismatch: {col_sums} vs {mu_t}"
#
#     print("Sinkhorn unit test passed successfully.")
#
# # Run the unit test
# test_sinkhorn()

import torch
import numpy
np = numpy
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss


# preferences, need to be converted to costs
# row i = cost of moving each item from c to place i
# making cost non-negative will not change solution matrix P
preference = numpy.asarray([[2, 2, 1 , 0 ,0],
                            [0,-2,-2,-2,  2],
                            [1, 2, 2, 2, -1],
                            [2, 1, 0, 1, -1],
                            [0.5, 2, 2, 1, 0],
                            [0,  1,1, 1, -1],
                            [-2, 2, 2, 1, 1],
                            [2, 1, 2, 1, -1]])

# how much do we have place available at place
r = (3,3,3,4,2,2,2,1)
r = torch.from_numpy(numpy.asarray(r)).float()

# how much do we need to transfer from each place
c = (4,2,6,4,4)
c = torch.from_numpy(numpy.asarray(c)).float()
x = torch.from_numpy(preference).float()


# from here https://michielstock.github.io/OptimalTransport/
def compute_optimal_transport(M, r, c, lam, epsilon=1e-8):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter
    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = torch.exp(- lam * M)
    P = P / P.sum()
    u = torch.ones(n)

    # normalize this matrix
    i = 0
    while torch.max(torch.abs(u - P.sum(dim=1))) > epsilon:
        u = P.sum(dim=1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
        i += 1
    print(i)
    return P, torch.sum(P * M)

def optimal_transport(M, r, c, lam, epsilon=1e-8):
    n, m = M.shape
    Kinit = torch.exp(-M * lam)
    K = torch.diag(1./r).mm(Kinit)

    # somehow faster
    u = r
    v = c
    vprev = v * 2
    i = 0
    while(torch.abs(v - vprev).sum() > epsilon):
        vprev = v
        # changing order affects convergence a little bit
        v = c / K.T.matmul(u)
        u = r / K.matmul(v)
        i += 1

    print(i)
    P = torch.diag(u) @ K @ torch.diag(v)
    return P, torch.sum(P * M)

# see https://arxiv.org/pdf/1612.02273.pdf
# https://arxiv.org/pdf/1712.03082.pdf
# but instead i multiply by lam like in code above
def optimal_transport_np(M, r, c, lam, epsilon=1e-8):
    n, m = M.shape
    Kinit = np.exp(- M * lam)
    K = np.diag(1./r).dot(Kinit)
    u = r
    v = c
    vprev = v * 2
    i = 0
    while(np.abs(v - vprev).sum() > epsilon):
        vprev = v
        v = c / K.T.dot(u)
        u = r / K.dot(v)
        i += 1
    print(i)
    P = np.diag(u) @ K @ np.diag(v)
    return P, np.sum(P * M)

def compute_optimal_transport_v2(M, mu_s, mu_t, num_iters=50, epsilon=1e-6):
    lam = 5
    n, m = M.shape

    Kinit = torch.unsqueeze(torch.exp(- M * lam), dim=0)
    temp = torch.diag_embed(1. / mu_s)
    K = torch.bmm(temp,Kinit)

    v = mu_t  # Shape: (B, num_proxies)
    u = mu_s  # Shape: (B, HW)

    i = 0

    for _ in range(num_iters):
        vprev = v.clone()
        v = mu_t / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, num_proxies)
        u = mu_s / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + epsilon)  # Shape: (B, HW)

        if torch.abs(v - vprev).sum() < epsilon:
            break

        i += 1
    print(i)
    T = torch.bmm(torch.bmm(torch.diag_embed(u), K), torch.diag_embed(v))  # Shape: (B, HW, num_proxies)
    return T, torch.sum(T * M)

P, cost = compute_optimal_transport(x * -1, r, c, 5)
print(P, cost)

P, cost = optimal_transport(x * -1, r, c, 5)
print(P, cost)

n, m = x.shape
r2 = torch.unsqueeze(r,dim=0)
c2 = torch.unsqueeze(c,dim=0)

P, cost = compute_optimal_transport_v2(x * -1, r2, c2, num_iters=100, epsilon=1e-9)
print(P, cost)

# shifting cost above zero will not change the solution P
x = x * -1
x = x - x.min()
P, cost = optimal_transport_np(x.numpy(), r.numpy(), c.numpy(), 5)
print(P, cost)
