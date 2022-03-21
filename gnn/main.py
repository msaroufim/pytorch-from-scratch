"""
Implementation copied and then modified with comments and types
from The Sebastian Raschka Book on
Machine Learning with sci-kit learn and PyTorch
"""

import torch
from typing import List
from logging import root
from torch.utils.data impor
from xml.sax.handler import property_dom_nodet Dataset
from torch import nn
from torch.nn.parameter import Parameter
import torch

class GraphNodeNetwork(nn.Module):
    """
    This is an example of a Graph Convolutional Network for graph classification
    It's called a Convolutional Network because it has a convolution layer that aggregates
    over neighbors and a pooling layer that aggregates the convolved neighbors
    The simplest way to aggregate is to just sum up the representations hence the global_add_pool()
    function
    """
    def __init__(self, input_features : int):
        super().__init__()
        # First Convolutional Layer
        self.conv1 = GraphConvolutionLayer(input_features, 32)
        
        # Second Convolutional Layer
        self.conv2 = GraphConvolutionLayer(32, 32)

        # Downsample to just 16 outputs
        self.fc1 = nn.Linear(32,16)

        # Downsample to 2 outputs
        self.out = nn.Linear(16,2)
    
    def forward(self, X, A, batch_mat):
        """
        X is the tensor representation of a node X in a graph
        A is the tensor representation of the graph
        A forward pass in a GNN works by updating the representation of a node X
        by using the representations of its neighbors in A
        """
        # First non linearity
        x = nn.ReLU(self.conv1(X, A))

        # Second non linearity
        x = nn.ReLU(self.conv2(x, A))

        # Make sure all graphs are the same size in case input was not batched
        output = global_sum_pool(x, batch_mat)

        output = self.fc1(output)
        output = self.out(output)

        # Turn 2 outputs into a probability for a classification
        return nn.Softmax(output, dim=1)
    
    def global_sum_pool(X, batch_mat):
        if batch_mat is None or batch_mat.dim() == 1:
            # If input is not batched then we don't know what the right sized graph should be
            # So instead we sum up all the node representations of a graph with torch.sum
            # Then we return a single value by using unsqueeze(0)
            return torch.sum(X, dim=0).unsqueeze(0)
        else:
            # Otherwise if input is batched then they must all be the same shape
            # Since by default PyTorch tensors are not jagged
            return torch.mm(batch_mat, X)



class ExampleDataset(Dataset):
    def __init__(self, graph_list : List):
        self.graphs = graph_list
    
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W1 = Parameter(torch.rand(in_channels, out_channels, dtype=torch.float32))

        self.W2 = Parameter(torch.rand(in_channels, out_channels, dtype=torch.float32))

        self.bias = Parameter(torch.zeros(out_channels))
    
    def forward(self, X, A):
        # Update current node X with matrix W1 
        X_update = torch.mm(X, self.W1)

        # Update with neighbor representation
        potential_msgs = torch.mm(X, self.W2)
        propagated_mgs = torch.mm(A, potential_msgs)

        # Sum up representations, you can come up with different ways of doing this
        output = propagated_mgs + X_update + self.bias

        return output
