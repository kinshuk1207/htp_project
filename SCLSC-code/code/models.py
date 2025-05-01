# models part
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# The supervised contrastive loss
# class ContrastiveLoss(torch.nn.Module):

#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, x0, x1, y):
#         diff = x0 - x1

#         # I CHANGE TORCH.SUM AXIS
#         dist_sq = torch.sum(torch.pow(diff, 2), -1)
#         dist = torch.sqrt(dist_sq)

#         mdist = self.margin - dist
#         dist = torch.clamp(mdist, min=0.0)
#         loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
#         loss = torch.sum(loss) / 2.0 / x0.size()[0]

#         return loss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = 0.1

    def forward(self, x0, x1, y):
        # x0: [B, 1, D], x1: [C, D], y: [B, C] one-hot
        # 1) squeeze out the extra dim
        x = x0.squeeze(1)                    # [B, D]
        # 2) L2-normalize both cell embeddings and prototypes
        x_norm   = F.normalize(x,  dim=1)    # [B, D]
        p_norm   = F.normalize(x1, dim=1)    # [C, D]
        # 3) compute similarity matrix
        sim = torch.matmul(x_norm, p_norm.t())  # [B, C]
        sim = sim / self.temperature

        # 4) log-softmax across prototypes
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [B, C]

        # 5) mask out only the positive prototype per cell
        #    y is one-hot [B, C]
        loss = - (y * log_prob).sum(dim=1) / (y.sum(dim=1) + 1e-8)    # [B]

        # 6) average over the batch
        return loss.mean()




# The MLP encoder structure, Yusri's current implementation
class MLP(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim2)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.batchnorm1(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.batchnorm2(out)
        out = self.dropout2(out)

        out = self.linear3(out)
        return out

class MLP_nobatchnorm(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP_nobatchnorm, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim2)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.linear3(out)
        return out

class MLP_nodo(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP_nodo, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.linear2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_dim1)
        self.batchnorm2 = torch.nn.BatchNorm1d(hidden_dim2)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.batchnorm1(out)

        out = self.linear2(out)
        out = F.relu(out)
        out = self.batchnorm2(out)

        out = self.linear3(out)
        return out

# use the CNN module previously used
class CNN(nn.Module):
        def __init__(self, input_dim, kernel_size=7, dr=0.1):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv1d(1,16,kernel_size=kernel_size, stride=2)
                self.bn1 = nn.BatchNorm1d(16)
                self.conv2 = nn.Conv1d(16,32, kernel_size=kernel_size, stride=2)
                self.bn2 = nn.BatchNorm1d(32)
                self.relu = nn.ReLU()

                self.maxpool = nn.MaxPool1d(2)

                self.fc1 = nn.Linear(7936, 32)
                #self.fc1 = nn.Linear(4608, 1474)


        def forward(self, x):
                x = self.bn1(self.relu(self.conv1(x)))
                x = self.bn2(self.relu(self.conv2(x)))
                x = self.maxpool(x)

                x = self.fc1(torch.flatten(x, 1))

                return x



# Function to transform raw representation to embedded representation
def project(encoder, X, device, encoder_model):

    if encoder_model == "CNN":
        X = np.expand_dims(X, 1)

    with torch.no_grad():
        encoder.eval()
        if torch.is_tensor(X):
            X = X.to(device=device, dtype=torch.float32)
        else:
            X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        emb_X = encoder(X).detach().cpu().numpy()
    return emb_X


# early stopping criterion
class EarlyStopper:

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):

            self.counter += 1
            print("Counter = %d, Val loss = %f, threshold= %f " %
                  (self.counter, validation_loss,
                   self.min_validation_loss + self.min_delta))
            if self.counter >= self.patience:
                return True
        return False

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.2):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim1)
        self.bn1   = nn.BatchNorm1d(hidden_dim1)
        self.do1   = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2   = nn.BatchNorm1d(hidden_dim2)
        self.do2   = nn.Dropout(dropout)
        self.skip1 = nn.Linear(input_dim, hidden_dim2) if input_dim != hidden_dim2 else nn.Identity()
        self.fc3   = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.fc1(x)))
        out1 = self.do1(out1)
        out2 = F.relu(self.bn2(self.fc2(out1)))
        out2 = self.do2(out2)
        res  = self.skip1(x)
        out2 = out2 + res
        return self.fc3(out2)

class GeneTransformer(nn.Module):
    def __init__(self,
                 n_genes: int,
                 emb_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dim_ff: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        # project raw gene vector to embedding dim
        self.input_proj = nn.Linear(n_genes, emb_dim)
        # positional embedding (optional, here just a learned vector)
        self.pos_emb    = nn.Parameter(torch.zeros(1, emb_dim))
        # transformer encoder stack
        layer = nn.TransformerEncoderLayer(
            d_model = emb_dim,
            nhead   = n_heads,
            dim_feedforward = dim_ff,
            dropout = dropout,
            activation = "gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        # final MLP head to your desired out_dim
        self.output_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        """
        x: [B, n_genes] gene counts or log‐counts
        """
        # 1) project
        h = self.input_proj(x)               # [B, emb_dim]
        # 2) add a constant "positional" embedding
        h = h + self.pos_emb                 # broadcast [B, emb_dim]
        # 3) transformer expects [S, B, E], so treat genes as sequence length=1
        h = h.unsqueeze(0)                   # [1, B, emb_dim]
        # 4) encode
        h = self.transformer(h)              # [1, B, emb_dim]
        h = h.squeeze(0)                     # [B, emb_dim]
        # 5) final projection
        return self.output_proj(h)           # [B, emb_dim]


class SCLSCWithPrototypes(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, out_dim, n_classes):
        super().__init__()
        # instead of MLP or ResidualMLP, use GeneTransformer:
        self.encoder    = GeneTransformer(
            n_genes = input_dim,
            emb_dim = out_dim,
            n_heads = 4,
            n_layers= 2,
            dim_ff   = hidden_dim2,
            dropout  = 0.1
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, out_dim))

    def forward(self, x):
        return self.encoder(x)

    def get_prototypes(self):
        return self.prototypes