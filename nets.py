import numpy as np
import torch
import torch.nn as nn

def fc_block(in_features, out_features, dropout_prob=None):
    layers = [nn.Linear(in_features, out_features)]
    if dropout_prob:
        layers.append(nn.Dropout(p=dropout_prob))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def fc_net(num_layers, in_size, hidden_size, out_proj_size=None, dropout_prob=None):
    layers = [
        fc_block(in_size if l == 0 else hidden_size, hidden_size, dropout_prob)
        for l in range(num_layers)]
    if out_proj_size is not None:
        layers.append(nn.Linear(hidden_size, out_proj_size))
    return nn.Sequential(*layers)

class CNNEncoder(nn.Module):
    def __init__(self, img_dim_hw, output_dim, embed_dim=128, stride=2, kernel_size=4, num_conv_layers=4):
        super(CNNEncoder, self).__init__()
        # Should be able to get exactly to the desired output size with some number of layers
        # Assumes stride=2, kernel_size=4, padding=1
        assert img_dim_hw[0] % 2**num_conv_layers == 0
        assert img_dim_hw[1] % 2**num_conv_layers == 0
        self.layers = []
        self.layers.append(nn.Conv2d(3, embed_dim, kernel_size, stride=stride, padding=1))
        self.layers.append(nn.ReLU())
        for _ in range(num_conv_layers - 1):
            self.layers.append(torch.nn.Conv2d(embed_dim, embed_dim, kernel_size, stride=stride, padding=1))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(embed_dim, output_dim))
        self.layers.append(nn.ReLU())
        
        self.layers = nn.ModuleList(self.layers)
    
    # input \in B x T x H x W x 3
    # output \in B x T x U x V x E
    def forward(self, x):
        batch_size = x.shape[0]
        num_frames = x.shape[1]

        # nn.Conv2d expects B x C x H x W, so collapse frames and batches, and swap channel
        x = x.flatten(0, 1)  # B*T x H x W x C
        
        x = x.moveaxis(-1, 1)  # B*T x C x H x W
        for layer in self.layers[:-2]:
            new_x = layer(x)
            x = new_x

        # Move back from ((B*T) x C x P x P) to (B x T x P x P x C)
        x = x.moveaxis(1, -1)  # B*T x H x W x C
        x = x.reshape((batch_size, num_frames) + x.shape[1:])
        for layer in self.layers[-2:]:
            x = layer(x)

        return x

# Input should have the format seq_len x batch_size x d_model
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = nn.functional.relu, layer_norm_eps: float = 1e-5, norm_first: bool = False) -> None:
        super(TransformerBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            if activation == 'relu':
                self.activation = nn.functional.relu
            elif activation == 'gelu':
                self.activation = nn.functional.gelu
            else:
                raise ValueError("activation should be relu/gelu, not {}".format(activation))
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class QueryDecoder(nn.Module):
    def __init__(
        self, input_size, query_size, hidden_size, output_size, num_hidden_layers,
        query_scale=1.0, output_scale=1.0):
        super(QueryDecoder, self).__init__()

        # we handle arbitrary output shapes, or just a scalar number of output dimensions
        output_size = [output_size] if np.isscalar(output_size) else output_size
        self.output_size = output_size
        num_output_dims = np.prod(output_size)

        # add the main fully connected layers
        next_input_size = input_size + query_size
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(fc_block(next_input_size, hidden_size))
            next_input_size = hidden_size
        self.hiddens = nn.Sequential(*hidden_layers)

        # final linear projection to the target number of output dimensions
        self.final_project = nn.Linear(next_input_size, num_output_dims)
        # register buffers to store the query and output scale factors
        self.register_buffer('query_scale', torch.tensor(query_scale))
        self.register_buffer('output_scale', torch.tensor(output_scale))

    def forward(self, z, query):
        # z.shape (B, ..., L), query.shape (B, ..., N, Q)
        query = query * self.query_scale
        num_queries = query.shape[-2]
        # replicate z over all queries (B, ..., L) --> (B, ..., N, L)
        z_query_tiled = torch.stack([z]*num_queries, -2)
        out = torch.cat([z_query_tiled, query], -1)  # concatenate z and query on last axis
        out = self.hiddens(out)  # apply main hidden layers
        out = self.final_project(out)  # apply final output projection
        out = out.unflatten(-1, self.output_size)  # reshape last axis to target output_size
        return out * self.output_scale