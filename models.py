"""Modules providing torch e dgl using torch"""
from dgl.nn.pytorch.conv import GraphConv, SAGEConv,GATv2Conv
from dgl.nn.pytorch import HeteroGraphConv, RelGraphConv,GINConv
from torch import nn
import torch.nn.functional as F
import torch


class GCN(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
        Input feature size.
    h_feats : int
        Hidden layers size.
    out_feats : int
        Output feature size.
    """
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        """Forward"""
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

##Multirelational aggregating multiple relations
class RGCN(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
        Input feature size.
    h_feats : int
        Hidden layers size.
    out_feats : int
        Output feature size.
    rel_names : list of string
       List of relation names
    """
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.name="RGCN"

        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        """Forward"""
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

#Aggregator type to use:(mean, gcn, pool, lstm)
class SAGE_RGCN(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
        Input feature size.
    h_feats : int
        Hidden layers size.
    out_feats : int
        Output feature size.
    rel_names : list of string
       List of relation names
    """
    def __init__(self, in_feats, hid_feats, out_feats, rel_names,aggregator_type="mean",p=0):
        super().__init__()
        self.name="SAGE_RGCN"
        self.dropout = nn.Dropout(p=p)
        self.conv1 = HeteroGraphConv({
            rel: SAGEConv(in_feats, hid_feats,aggregator_type=aggregator_type)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = HeteroGraphConv({
            rel: SAGEConv(hid_feats, out_feats,aggregator_type=aggregator_type)
            for rel in rel_names}, aggregate='sum')
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv1.mods.values():
            conv.reset_parameters()
        for conv in self.conv2.mods.values():
            conv.reset_parameters()

    def forward(self, graph, inputs):
        """Forward"""
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
    
class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 =GATv2Conv(in_feats, hid_feats, num_heads=4)
        self.conv2 =GATv2Conv(hid_feats*4, out_feats, num_heads=1)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)        
        h = F.relu(h) 
        h = self.conv2(graph, h)
        return h

class RecurrentHeteroGraphNN(nn.Module):
    """RecurrentHeteroGraphNN"""
    def __init__(self, in_feats, hidden_feats, out_feats, rel_names, num_layers,aggregate="sum"):
        super(RecurrentHeteroGraphNN, self).__init__()
        self.name="RecurrentGNN"
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.recurrent = nn.ModuleList()
        self.hidden_feats=hidden_feats
        
        # Primo strato di convoluzione
        self.convs.append(HeteroGraphConv({
            rel: GraphConv(in_feats, hidden_feats)
            for rel in rel_names}, aggregate=aggregate))
        
        # Strati ricorrenti
        for _ in range(num_layers - 1):
            self.convs.append(HeteroGraphConv({
                rel: GraphConv(hidden_feats, hidden_feats)
                for rel in rel_names}, aggregate=aggregate))
            self.recurrent.append(nn.GRU(hidden_feats, hidden_feats))
        
        # Strato finale di convoluzione
        self.convs.append(HeteroGraphConv({
            rel: GraphConv(hidden_feats, out_feats)
            for rel in rel_names}, aggregate=aggregate))
        
    def forward(self, g, features, hidden):
        h = features
        for i in range(self.num_layers - 1):
            h = self.convs[i](g, h)
            h = {k: F.relu(v) for k, v in h.items()}
            # Concatenazione delle feature dei nodi per tutti i tipi di nodi
            h_cat = torch.cat([h[ntype] for ntype in h.keys()], dim=0)
            #unsqueeze aggiunge una dimensione al posto 0 
            h_cat, hidden[i] = self.recurrent[i](h_cat.unsqueeze(0), hidden[i])
            #squeeze toglie una dimensione al posto 0 
            h_cat = h_cat.squeeze(0)
           
            # Dividere h_cat per ogni tipo di nodo
            idx = 0
            new_h = {}
            for ntype in h.keys():
                new_h[ntype] = h_cat[idx:idx + h[ntype].shape[0], :]
                idx += h[ntype].shape[0]
            h = new_h
        
        h = self.convs[-1](g, h)
        return h
    def init_hidden(self, batch_size, device):
        # Inizializzazione degli stati nascosti
        hidden = []
        for _ in range(self.num_layers - 1):
            hidden.append(torch.zeros(1, batch_size, self.hidden_feats).to(device))
        return hidden

class MLP(nn.Module):
    """MLP for GIN"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.01))
        layers.append(nn.Dropout(p=dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            #Ogni layer nella MLP ha un layer di dropout per regolarizzare il modello e ridurre la sparseness.
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class GIN(nn.Module):
    """GIN: Graph Isomorphism Network"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, learn_eps=False, dropout=0.2):
        super(GIN, self).__init__()
        self.name="GIN"
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim, dropout)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, dropout)
            self.gin_layers.append(GINConv(mlp, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, g, features):
        h = features
        for i in range(self.num_layers):
            h = self.gin_layers[i](g, h)
            h = self.batch_norms[i](h)
            if i != self.num_layers - 1:  # Apply activation only to intermediate layers
                h = F.leaky_relu(h, negative_slope=0.01)
        return h
        