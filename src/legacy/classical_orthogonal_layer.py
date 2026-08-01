"""Original sparse-COO orthogonal layer preserved verbatim for benchmarking."""

import numpy as np
import torch


class LegacyOrthoLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)

        size = (2 * larger_features - 1 - smaller_features) * smaller_features // 2
        self.thetas = torch.nn.Parameter(torch.randn(int(size)))
        self.bias = torch.nn.Parameter(torch.zeros(int(out_features)))

        if larger_features == smaller_features:
            smaller_features -= 1

        self.x_end_index = np.concatenate([
            np.arange(2, larger_features + 1),
            larger_features + 1 - np.arange(2, smaller_features + 1),
        ])
        self.x_start_index = np.concatenate([
            np.arange(self.x_end_index.shape[0] + smaller_features - larger_features) % 2,
            np.arange(larger_features - smaller_features),
        ])
        self.x_slice_sizes = self.x_end_index - self.x_start_index

        self.precomputed_indices = []
        for slice_size in self.x_slice_sizes:
            n = slice_size // 2
            if n == 0:
                self.precomputed_indices.append(None)
                continue
            row_idx = torch.cat([
                torch.tensor([2 * i, 2 * i, 2 * i + 1, 2 * i + 1])
                for i in range(n)
            ])
            col_idx = torch.cat([
                torch.tensor([2 * i, 2 * i + 1]).repeat(2)
                for i in range(n)
            ])
            self.precomputed_indices.append(torch.stack([row_idx, col_idx]))

    def hidden_layer(self, x, in_features, out_features):
        if in_features < out_features:
            x_end_index = self.x_end_index[::-1]
            x_start_index = self.x_start_index[::-1]
            x_slice_sizes = self.x_slice_sizes[::-1]
            precomputed_indices = self.precomputed_indices[::-1]
            x = torch.nn.functional.pad(x, (out_features - in_features, 0))
        else:
            x_end_index = self.x_end_index
            x_start_index = self.x_start_index
            x_slice_sizes = self.x_slice_sizes
            precomputed_indices = self.precomputed_indices

        theta_start = 0
        for i, sz in enumerate(x_slice_sizes):
            n = sz // 2
            if n == 0:
                continue
            theta_end = theta_start + n
            theta_slice = self.thetas[theta_start:theta_end]
            theta_start = theta_end
            x_slice = x[:, x_start_index[i]:x_end_index[i]]
            cos_t = torch.cos(theta_slice)
            sin_t = torch.sin(theta_slice)
            values = torch.stack((cos_t, sin_t, -sin_t, cos_t), dim=1).view(-1)
            indices = precomputed_indices[i].to(x.device)
            rotation = torch.sparse_coo_tensor(indices, values, (2 * n, 2 * n))
            x_new = x.clone()
            x_new[:, x_start_index[i]:x_end_index[i]] = torch.mm(x_slice, rotation)
            x = x_new

        if in_features > out_features:
            x = x[:, in_features - out_features:]
        return x + self.bias

    def forward(self, x):
        if x.shape[1] != self.in_features:
            raise AssertionError(f"x shape {x.shape} isn't equal to {self.in_features}")
        return self.hidden_layer(x, self.in_features, self.out_features)
