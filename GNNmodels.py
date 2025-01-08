import torch


import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as datasets

from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    
  def __init__(
      self,
      input_dim: int,
      hid_dim: int,
      n_classes: int,
      n_layers: int,
      dropout_ratio: float = 0.3):
    """
    Args:
      input_dim: input feature dimension
      hid_dim: hidden feature dimension
      n_classes: number of target classes
      n_layers: number of layers
      dropout_ratio: dropout_ratio
    """
      
    super(GCN, self).__init__()
    
    self.input_dim=input_dim
    self.hid_dim=hid_dim
    self.n_classes=n_classes
    self.n_layers=n_layers
    self.dropout_ratio=dropout_ratio
    self.param_init()

  def forward(self, X, A) -> torch.Tensor:
      
    if len(self.convs) == 0:  # If layers=0, feedforward network
      return self.feed_forward(X)
    else:
      for i, conv in enumerate(self.convs):
          X = conv(X, A)
          if i< len(self.convs) - 1:
            X = F.relu(X)
            X = self.dropout(X)
      return X

  def generate_node_embeddings(self, X, A) -> torch.Tensor:
    with torch.no_grad():
      if len(self.convs) == 0:  # If layers=0, feedforward network
        return self.feed_forward(X)
      else:
      #No dropout
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                X = conv(X, A)
                X = F.relu(X)
        return X
      
  def generate_node_embeddings_perlayer(self, X, A) -> torch.Tensor:

    with torch.no_grad():
      embeddings=[]
      if len(self.convs) == 0:  # If layers=0, feedforward network
        return self.feed_forward(X)
      else:
      #No dropout
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                X = conv(X, A)
                embeddings.append(X)
                X = F.relu(X)
        return embeddings

  def param_init(self):
      
    #In case n_layers=0
    if self.n_layers==0:
        self.feed_forward = nn.Linear(self.input_dim, self.n_classes)

    else:
        # Define the first GCN layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.input_dim, self.hid_dim))

        # Define intermediate layers
        for _ in range(self.n_layers - 2):
            self.convs.append(GCNConv(self.hid_dim, self.hid_dim))

        # Define the final GCN layer
        self.convs.append(GCNConv(self.hid_dim, self.n_classes))

        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_ratio)
 