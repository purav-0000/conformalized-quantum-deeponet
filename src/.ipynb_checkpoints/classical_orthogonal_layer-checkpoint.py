import torch
import numpy as np


class OrthoLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)
        size = (2 * larger_features - 1 - smaller_features) * smaller_features / 2  # number of free parameters
        # torch.manual_seed(0)
        self.thetas = torch.nn.Parameter(torch.randn(int(size)))  # normal distribution initializer for thetas
        self.bias = torch.nn.Parameter(torch.zeros(int(out_features)))

        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)

        if larger_features == smaller_features:
            smaller_features -= 1  # 6-6 6-5 have the same pyramid
        self.x_end_index = np.concatenate([
            np.arange(2, larger_features + 1),
            larger_features + 1 - np.arange(2, smaller_features + 1)
        ])
        self.x_start_index = np.concatenate([
            np.arange(self.x_end_index.shape[0] + smaller_features - larger_features) % 2,  # [0, 1, 0, 1, ...]
            np.arange(larger_features - smaller_features)
        ])

        self.x_slice_sizes = self.x_end_index - self.x_start_index

        if in_features < out_features:
            self.x_end_index = self.x_end_index[::-1]
            self.x_start_index = self.x_start_index[::-1]
            self.x_slice_sizes = self.x_slice_sizes[::-1]

        self.indices = []

        theta_start_index = 0
        for i in range(len(self.x_start_index)):
            n = self.x_slice_sizes[i] // 2 - theta_start_index
            self.indices.append(torch.stack([
                torch.cat([torch.tensor([2 * i, 2 * i, 2 * i + 1, 2 * i + 1]) for i in range(n)]),
                torch.cat([torch.tensor([2 * i, 2 * i + 1]).repeat(2) for i in range(n)])
            ]))



    def hidden_layer(self, x, in_features, out_features):

        if in_features < out_features:  # generate the pyramid for in_features < out_features case
            x = torch.nn.functional.pad(x, (out_features - x.shape[1], 0))  # pad x fist if in_features < out_features case

        theta_start_index = 0

        for i in range(len(self.x_start_index)):
            theta_slice = self.thetas[theta_start_index:theta_start_index + self.x_slice_sizes[i] // 2]
            theta_start_index = theta_start_index + self.x_slice_sizes[i] // 2
            x_slice = x[:, self.x_start_index[i]:self.x_end_index[i]]

            # generate rotation matrix
            n = len(theta_slice)
            theta_slice = theta_slice.view(-1, 1)
            values = torch.cat(
                [torch.cos(theta_slice), torch.sin(theta_slice), -torch.sin(theta_slice), torch.cos(theta_slice)],
                dim=1).view(-1)
            rotation_matrix = torch.sparse_coo_tensor(self.indices[i], values, size=[2 * n, 2 * n])

            x = x.clone()
            x[:, self.x_start_index[i]:self.x_end_index[i]] = torch.mm(x_slice, rotation_matrix)

        if in_features > out_features:
            x = x[:, in_features - out_features:]

        return x + self.bias

    def forward(self, x):
        if x.shape[1] != self.in_features:
            raise AssertionError(
                f'x shape {x.shape} isn\'t equal to {self.in_features}'
            )
        x = self.hidden_layer(x, self.in_features, self.out_features)
        return x