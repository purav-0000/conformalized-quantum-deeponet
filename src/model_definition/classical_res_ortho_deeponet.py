import os
os.environ['DDE_BACKEND'] = 'pytorch'

import torch
import deepxde as dde

# COMMENTED THE LINE BELOW OUT AND INSTEAD IMPORTED TORCH
#from deepxde.backend import torch
import torch

import deepxde.nn.activations as activations

# ADDED SRC. TO SRC.CLASSICAL_ORTHOGONAL_LAYER
from src.model_definition.classical_orthogonal_layer import OrthoLayer

class ResOrthoNN(dde.nn.pytorch.NN): 
    def __init__(self, layer_sizes, activation):  #e.g. [3,4,5,1] need data pre-processing
        super().__init__()
        # determine the activation function
        self.activation = activations.get(activation)
        self.layer_sizes = layer_sizes
        self.hidden_layers= torch.nn.ModuleList()
        # self.dropout_rate = dropout_rate # ADDED: Store dropout_rate

        # ADDED: Create a separate ModuleList for dropout layers
        # self.dropouts = torch.nn.ModuleList()

        for i in range(len(layer_sizes)-2):
            self.hidden_layers.append(OrthoLayer(layer_sizes[i],layer_sizes[i+1]))
            # ADDED: Append a dropout layer for each hidden layer
            # self.dropouts.append(torch.nn.Dropout(p=self.dropout_rate))
        self.output_layer = torch.nn.Linear(layer_sizes[-2],layer_sizes[-1])        
    
    def forward(self,inputs):
        x = inputs

        # apply input transform when input is 1-dimention
        if self._input_transform is not None:
            x = self._input_transform(x)
    
        for i in range(len(self.layer_sizes)-2):
            norm = x.norm(dim=1, keepdim=True)
            x = x/norm
            if i == 0: 
                x = self.activation(self.hidden_layers[i](x))
            else:
                x =  self.activation(self.hidden_layers[i](x))+x

            # ADDED: Apply dropout after activation
            # x = self.dropouts[i](x)
        
        #res = x    
        x = self.output_layer(x) 
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    
# define ResNet architecture
class ResOrthoONetCartesianProd(dde.nn.pytorch.NN):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        regularization=None,
        online=False
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)

        self.branch = ResOrthoNN(layer_sizes_branch, activation_branch)
        self.trunk = ResOrthoNN(layer_sizes_trunk, self.activation_trunk)
        self.b = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.regularizer = regularization
        self._branch_transform = None
        self._trunk_transform = None
        self.online = online

    #input tranformation
    def apply_branch_transform(self, transform):
        self._branch_transform = transform
    def apply_trunk_transform(self, transform):
        self._trunk_transform = transform

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        if self._branch_transform is not None:
            x_func = self._branch_transform(x_func)
        # Branch net to encode the input function
        x_func = self.branch(x_func)

        if self._trunk_transform is not None:
            x_loc = self._trunk_transform(x_loc)
        x_loc = self.trunk(x_loc)
        x_loc = self.activation_trunk(x_loc)
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )

        # Evaluate batch with corresponding batch if online
        if self.online:
            x = torch.einsum("bi,bi->b", x_func, x_loc)
        else:
            x = torch.einsum("bi,ni->bn", x_func, x_loc)

        # Add bias
        x += self.b

        # Batch-to-batch einsum is flattening the array I think
        return x if not self.online else x.reshape(-1, 1)

class ResNN(dde.nn.pytorch.NN): 
    def __init__(self, layer_sizes, activation):  #e.g. [3,4,5,1] need data pre-processing
        super().__init__()
        # determine the activation function
        self.activation = activations.get(activation)
        self.layer_sizes = layer_sizes
        self.hidden_layers= torch.nn.ModuleList()

        for i in range(len(layer_sizes)-2):
            self.hidden_layers.append(torch.nn.Linear(layer_sizes[i],layer_sizes[i+1]))
        self.output_layer = torch.nn.Linear(layer_sizes[-2],layer_sizes[-1])
    
    def forward(self,inputs):
        x = inputs

        # apply input transform when input is 1-dimention
        if self._input_transform is not None:
            x = self._input_transform(x)
    
        for i in range(len(self.layer_sizes)-2):
            #! do not normalize the input
            # norm = x.norm(dim=1, keepdim=True)
            # x = x/norm
            if i == 0: 
                x = self.activation(self.hidden_layers[i](x))
            else:
                x =  self.activation(self.hidden_layers[i](x))+x

        #res = x    
        x = self.output_layer(x) 
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    
class ResONetCartesianProd(dde.nn.pytorch.NN):
    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        regularization=None,
        online=False
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)

        self.branch = ResNN(layer_sizes_branch, activation_branch)
        self.trunk = ResNN(layer_sizes_trunk, self.activation_trunk)
        self.b = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        self.regularizer = regularization
        self._branch_transform = None
        self._trunk_transform = None
        self.online = online

    #input tranformation
    def apply_branch_transform(self, transform):
        self._branch_transform = transform
    def apply_trunk_transform(self, transform):
        self._trunk_transform = transform

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        if self._branch_transform is not None:
            x_func = self._branch_transform(x_func)
        # Branch net to encode the input function
        x_func = self.branch(x_func)

        if self._trunk_transform is not None:
            x_loc = self._trunk_transform(x_loc)
        x_loc = self.trunk(x_loc)
        x_loc = self.activation_trunk(x_loc)
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )

        # Evaluate batch with corresponding batch if online
        if self.online:
            x = torch.einsum("bi,bi->b", x_func, x_loc)
        else:
            x = torch.einsum("bi,ni->bn", x_func, x_loc)

        # Add bias
        x += self.b

        # Batch-to-batch einsum is flattening the array I think
        return x if not self.online else x.reshape(-1, 1)
    