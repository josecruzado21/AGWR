import torch
from torch_geometric.nn import RGATConv

class ContextRGAT(torch.nn.Module):
    """
    Contextual Relational Graph Attention Network (ContextRGAT) model.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden layer dimension.
        out_dim (int): Output feature dimension.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = RGATConv(in_dim, hidden_dim, num_relations=3, edge_dim=1)
        self.conv2 = RGATConv(hidden_dim, out_dim, num_relations=3, edge_dim=1)
        
    def forward(self, x, edge_index, edge_type, edge_attr):
        """
        Forward pass of the ContextRGAT model.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph edge indices.
            edge_type (torch.Tensor): Edge types.
            edge_attr (torch.Tensor): Edge attributes.

        Returns:
            torch.Tensor: Output node features.
        """
        x = self.conv1(x, edge_index, edge_type, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_type, edge_attr)
        return x