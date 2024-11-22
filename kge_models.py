"""Importing torch modules"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""La classe Embedding è un semplice modo di immagazzinare gli embedding. Questi si possono ottenere
richiamando la variabile 
weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from Normal Distribution N(0,1) """

class TransE(nn.Module):
    """
    TransE
    Parameters
    ----------
    num_nodes : int
        Input total number of nodes.
    num_relations : int
        Input number of unique relations.
    hidden_channels : int
        Output embedding size.
    """
    def __init__(self, num_nodes, num_relations, hidden_channels,use_init_embeddings,init_ent_embs):
        super(TransE, self).__init__()
        if use_init_embeddings:
            self.entity_embedding = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
        else:
            self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        #self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        self.relation_embedding = nn.Embedding(num_relations, hidden_channels)

    def forward(self, src, rel, dst):
        src_emb = self.entity_embedding(src)
        dst_emb = self.entity_embedding(dst)
        rel_emb = self.relation_embedding(rel)
        
        # Calcola la differenza tra l'embedding della sorgente e della destinazione
        # e applica la relazione (embedding della relazione) come traslazione
        src_emb = F.normalize(src_emb, p=2, dim=-1)
        dst_emb = F.normalize(dst_emb, p=2, dim=-1)
        score = torch.norm(src_emb + rel_emb - dst_emb, dim=-1, p=2)
        
        return score
class TransR(nn.Module):
    """
    TransR
    Parameters
    ----------
    num_nodes : int
        Input total number of nodes.
    num_relations : int
        Input number of unique relations.
    hidden_channels : int
        Output embedding size.
    r_dim :  int
        Dimension of relations space
    """
    def __init__(self, num_nodes, num_relations, hidden_channels, r_dim,use_init_embeddings,init_ent_embs):
        super(TransR, self).__init__()
        if use_init_embeddings:
            self.entity_embedding = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
        else:
            self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        #self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        self.relation_embedding = nn.Embedding(num_relations, r_dim)
        self.projection_matrix = nn.Parameter(torch.randn(num_relations, hidden_channels, r_dim))
        self.hidden_channels = hidden_channels
        self.r_dim = r_dim

    def forward(self, src, rel, dst):
        src_emb = self.entity_embedding(src)
        dst_emb = self.entity_embedding(dst)
        rel_emb = self.relation_embedding(rel)
        proj_matrix = self.projection_matrix[rel]

        # Proiezione degli embedding delle entità nello spazio della relazione
        src_proj = torch.matmul(src_emb.unsqueeze(1), proj_matrix).squeeze(1)
        dst_proj = torch.matmul(dst_emb.unsqueeze(1), proj_matrix).squeeze(1)
        
        src_proj = F.normalize(src_proj, p=2, dim=-1)
        dst_proj = F.normalize(dst_proj, p=2, dim=-1)
        # Calcola la differenza tra l'embedding della sorgente proiettata e della destinazione proiettata
        # e applica la relazione (embedding della relazione) come traslazione
        score = torch.norm(src_proj + rel_emb - dst_proj, dim=-1, p=2)

        return score

class DistMult(nn.Module):
    """
    DistMult
    Parameters
    ----------
    num_nodes : int
        Input total number of nodes.
    num_relations : int
        Input number of unique relations.
    hidden_channels : int
        Output embedding size.
    """
    def __init__(self, num_nodes, num_relations, hidden_channels,use_init_embeddings=False,init_ent_embs=None):
        super(DistMult, self).__init__()
        if use_init_embeddings:
            self.entity_embedding = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
        else:
            self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        self.relation_embedding = nn.Embedding(num_relations, hidden_channels)

    def forward(self, src, rel, dst):
        src_emb = self.entity_embedding(src)
        dst_emb = self.entity_embedding(dst)
        rel_emb = self.relation_embedding(rel)
        
        score = torch.sum(src_emb * rel_emb * dst_emb, dim=-1)
        
        return score

class ComplEx(nn.Module):
    """
    ComplEx
    Parameters
    ----------
    num_nodes : int
        Input total number of nodes.
    num_relations : int
        Input number of unique relations.
    hidden_channels : int
        Output embedding size.
    """
    def __init__(self, num_nodes, num_relations, hidden_channels,use_init_embeddings=False,init_ent_embs=None):
        super(ComplEx, self).__init__()
        
        if use_init_embeddings:
            self.entity_embedding_real = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
            self.entity_embedding_imag = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
        else:
            self.entity_embedding_real = nn.Embedding(num_nodes, hidden_channels)
            self.entity_embedding_imag = nn.Embedding(num_nodes, hidden_channels)
        
        self.relation_embedding_real = nn.Embedding(num_relations, hidden_channels)
        self.relation_embedding_imag = nn.Embedding(num_relations, hidden_channels)
        self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        with torch.no_grad():
            self.entity_embedding.weight.copy_(self.entity_embedding_real.weight + self.entity_embedding_imag.weight)
    def forward(self, src, rel, dst):
        src_emb_real = self.entity_embedding_real(src)
        src_emb_imag = self.entity_embedding_imag(src)
        dst_emb_real = self.entity_embedding_real(dst)
        dst_emb_imag = self.entity_embedding_imag(dst)
        rel_emb_real = self.relation_embedding_real(rel)
        rel_emb_imag = self.relation_embedding_imag(rel)

        score = torch.sum(
            src_emb_real * rel_emb_real * dst_emb_real +
            src_emb_real * rel_emb_imag * dst_emb_imag +
            src_emb_imag * rel_emb_real * dst_emb_imag -
            src_emb_imag * rel_emb_imag * dst_emb_real, dim=-1)
        
        return score
class SimplE(nn.Module):
    """
    SimplE
    Parameters
    ----------
    num_nodes : int
        Input total number of nodes.
    num_relations : int
        Input number of unique relations.
    hidden_channels : int
        Output embedding size.
    """
    def __init__(self, num_nodes, num_relations, hidden_channels,use_init_embeddings=False,init_ent_embs=None):
        super(SimplE, self).__init__()
        
        if use_init_embeddings:
            self.entity_embedding_real = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
            self.entity_embedding_imag = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
        else:
            self.entity_embedding_real = nn.Embedding(num_nodes, hidden_channels)
            self.entity_embedding_imag = nn.Embedding(num_nodes, hidden_channels)


        self.relation_embedding_real = nn.Embedding(num_relations, hidden_channels)
        self.relation_embedding_imag = nn.Embedding(num_relations, hidden_channels)
        # Calcolo e assegnazione degli embedding complessi delle entità come attributo entity_embedding
        self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        with torch.no_grad():
            self.entity_embedding.weight.copy_(self.entity_embedding_real.weight + self.entity_embedding_imag.weight)
            
    def forward(self, src, rel, dst):
        src_emb_real = self.entity_embedding_real(src)
        src_emb_imag = self.entity_embedding_imag(src)
        dst_emb_real = self.entity_embedding_real(dst)
        dst_emb_imag = self.entity_embedding_imag(dst)
        rel_emb_real = self.relation_embedding_real(rel)
        rel_emb_imag = self.relation_embedding_imag(rel)

        # Normalizzazione degli embedding delle entità per la parte reale e immaginaria separatamente
        src_emb_real = F.normalize(src_emb_real, p=2, dim=-1)
        src_emb_imag = F.normalize(src_emb_imag, p=2, dim=-1)
        dst_emb_real = F.normalize(dst_emb_real, p=2, dim=-1)
        dst_emb_imag = F.normalize(dst_emb_imag, p=2, dim=-1)

        score = torch.sum(
            src_emb_real * rel_emb_real * dst_emb_real +
            src_emb_imag * rel_emb_real * dst_emb_imag +
            src_emb_real * rel_emb_imag * dst_emb_imag +
            src_emb_imag * rel_emb_imag * dst_emb_real, dim=-1)
        
        return score
class RotatE(nn.Module):
    """
    RotatE
    Parameters
    ----------
    num_nodes : int
        Input total number of nodes.
    num_relations : int
        Input number of unique relations.
    hidden_channels : int
        Output embedding size.
    """
    def __init__(self, num_nodes, num_relations, hidden_channels, use_init_embeddings=False, init_ent_embs=None):
        super(RotatE, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        if use_init_embeddings:
            self.entity_embedding_real = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
            self.entity_embedding_imag = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
        else:
            self.entity_embedding_real = nn.Embedding(num_nodes, hidden_channels)
            self.entity_embedding_imag = nn.Embedding(num_nodes, hidden_channels)

        self.relation_embedding_real = nn.Embedding(num_relations, hidden_channels)
        self.relation_embedding_imag = nn.Embedding(num_relations, hidden_channels)

        # Calcolo e assegnazione degli embedding complessi delle entità come attributo entity_embedding
        self.entity_embedding = nn.Embedding(num_nodes, hidden_channels)
        with torch.no_grad():
            self.entity_embedding.weight.copy_(self.entity_embedding_real.weight + self.entity_embedding_imag.weight)
    def forward(self, src, rel, dst):
        src_emb_real = self.entity_embedding_real(src)
        src_emb_imag = self.entity_embedding_imag(src)
        dst_emb_real = self.entity_embedding_real(dst)
        dst_emb_imag = self.entity_embedding_imag(dst)
        rel_emb_real = self.relation_embedding_real(rel)
        rel_emb_imag = self.relation_embedding_imag(rel)
        
        # Normalizzazione degli embedding delle entità per la parte reale e immaginaria separatamente
        src_emb_real = F.normalize(src_emb_real, p=2, dim=-1)
        src_emb_imag = F.normalize(src_emb_imag, p=2, dim=-1)
        dst_emb_real = F.normalize(dst_emb_real, p=2, dim=-1)
        dst_emb_imag = F.normalize(dst_emb_imag, p=2, dim=-1)
        rel_emb_real = F.normalize(rel_emb_real, p=2, dim=-1)
        rel_emb_imag = F.normalize(rel_emb_imag, p=2, dim=-1)

        re_score = torch.sum(
            (src_emb_real * rel_emb_real - src_emb_imag * rel_emb_imag) * dst_emb_real +
            (src_emb_real * rel_emb_imag + src_emb_imag * rel_emb_real) * dst_emb_imag, dim=-1)
        im_score = torch.sum(
            (src_emb_real * rel_emb_real - src_emb_imag * rel_emb_imag) * dst_emb_imag +
            (src_emb_imag * rel_emb_real + src_emb_real * rel_emb_imag) * dst_emb_real, dim=-1)

        return torch.sqrt(re_score ** 2 + im_score ** 2)


# Definizione del modello ConvE
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ConvE(nn.Module):
    """Dimensione di output sarà embedding_size_h X embedding_size_w = 200"""
    def __init__(self, num_e, num_r, embedding_size_h=16, embedding_size_w=24,
                 conv_channels=32, conv_kernel_size=3, embed_dropout=0.2, feature_map_dropout=0.2,
                 proj_layer_dropout=0.3, use_init_embeddings=False, init_ent_embs=None):
        super().__init__()

        self.num_e = num_e
        self.num_r = num_r
        self.embedding_size_h = embedding_size_h
        self.embedding_size_w = embedding_size_w

        embedding_size = embedding_size_h * embedding_size_w
        flattened_size = (embedding_size_w * 2 - conv_kernel_size + 1) * \
                         (embedding_size_h - conv_kernel_size + 1) * conv_channels

        if use_init_embeddings:
            self.entity_embedding = nn.Embedding.from_pretrained(init_ent_embs,freeze=False)
        else:
            self.entity_embedding = nn.Embedding(num_embeddings=self.num_e, embedding_dim=embedding_size)
        
        self.embed_r = nn.Embedding(num_embeddings=self.num_r, embedding_dim=embedding_size)

        self.conv_e = nn.Sequential(
            nn.Dropout(p=embed_dropout),
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.Dropout2d(p=feature_map_dropout),

            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=embedding_size),
            nn.Dropout(p=proj_layer_dropout)
        )

    def forward(self, s, r, t=None):
        embed_s = self.entity_embedding(s)
        embed_r = self.embed_r(r)

        # Reshape embeddings for convolution
        embed_s = embed_s.view(-1, self.embedding_size_w, self.embedding_size_h)
        embed_r = embed_r.view(-1, self.embedding_size_w, self.embedding_size_h)
        
        # Concatenate and prepare for convolution
        conv_input = torch.cat([embed_s, embed_r], dim=1).unsqueeze(1)
        out = self.conv_e(conv_input)

        if t is not None:
            embed_t = self.entity_embedding(t)
            scores = torch.sum(out * embed_t, dim=1)
            return scores
        else:
            scores = out.mm(self.entity_embedding.weight.t())
            return scores
        
class ConvKB(nn.Module):

    def __init__(self, entTotal, relTotal, hidden_size, out_channels, kernel_size, use_init_embeddings=False, init_ent_embs=None, convkb_drop_prob=0.5, lmbda=0.01):
        super(ConvKB, self).__init__()
        self.hidden_size=hidden_size
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.ent_embeddings = nn.Embedding(entTotal, hidden_size) 
        self.rel_embeddings = nn.Embedding(relTotal, hidden_size)

        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1, out_channels, (kernel_size, 3))  # kernel size x 3
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(convkb_drop_prob)
        self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
        self.fc_layer = nn.Linear((hidden_size - kernel_size + 1) * out_channels, 1, bias=False)

        self.criterion = nn.Softplus()
        self.use_init_embeddings = use_init_embeddings
        if self.use_init_embeddings:
            self.init_ent_embs = init_ent_embs

        self.init_parameters()

    def init_parameters(self):
        if not self.use_init_embeddings:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        else:
            self.ent_embeddings.weight.data = self.init_ent_embs
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.fc_layer.weight.data)
        nn.init.xavier_uniform_(self.conv_layer.weight.data)

    def _calc(self, h, r, t):
        h = h.unsqueeze(1) # bs x 1 x dim
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)

        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.hidden_size - self.kernel_size + 1) * self.out_channels)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)
        
        return -score

    def forward(self, batch_h, batch_r, batch_t):
        h = self.ent_embeddings(batch_h)
        r = self.rel_embeddings(batch_r)
        t = self.ent_embeddings(batch_t)
        score = self._calc(h, r, t)

        return score
        
class ConvR(nn.Module):
    def __init__(self, num_e, num_r, embedding_size_h=20, embedding_size_w=10,
                 conv_channels=32, conv_kernel_size=3, embed_dropout=0.2, feature_map_dropout=0.2,
                 proj_layer_dropout=0.3):
        super().__init__()

        self.num_e = num_e
        self.num_r = num_r
        self.embedding_size_h = embedding_size_h
        self.embedding_size_w = embedding_size_w

        embedding_size = embedding_size_h * embedding_size_w
        flattened_size = (embedding_size_w * 2 - conv_kernel_size + 1) * \
                         (embedding_size_h - conv_kernel_size + 1) * conv_channels

        self.entity_embedding = nn.Embedding(num_embeddings=self.num_e, embedding_dim=embedding_size)
        self.relation_embedding = nn.Embedding(num_embeddings=self.num_r, embedding_dim=embedding_size)
        
        # Aggiunta della convoluzione relazionale
        self.conv_r = nn.Sequential(
            nn.Dropout(p=embed_dropout),
            nn.Conv2d(in_channels=2, out_channels=conv_channels, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=conv_channels),
            nn.Dropout2d(p=feature_map_dropout),
            Flatten(),
            nn.Linear(in_features=flattened_size, out_features=embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=embedding_size),
            nn.Dropout(p=proj_layer_dropout)
        )

    def forward(self, s, r, t=None):
        embed_s = self.entity_embedding(s)
        embed_r = self.relation_embedding(r)

        # Reshape degli embedding per la convoluzione
        embed_s = embed_s.view(-1, self.embedding_size_w, self.embedding_size_h)
        embed_r = embed_r.view(-1, self.embedding_size_w, self.embedding_size_h)
        
        # Concatenazione di s e r per formare l'input per la convoluzione relazionale
        conv_input = torch.cat([embed_s, embed_r], dim=1)
        out = self.conv_r(conv_input)

        if t is not None:
            embed_t = self.entity_embedding(t)
            scores = torch.sum(out * embed_t, dim=1)
            return scores
        else:
            scores = out.mm(self.entity_embedding.weight.t())
            return scores