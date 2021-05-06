import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def similarity(self, spatial_embedding):
        e0 = spatial_embedding.unsqueeze(2)
        e1 = spatial_embedding.unsqueeze(1)
        dist = (e0 - e1).norm(2, dim=-1)
        sim = (-dist.pow(2)).exp()
        sim = sim / sim.sum(dim=-1, keepdims=True)
        return sim

    def forward(self, spatial_embedding, z):
        # The relation to Attention is as follows:
        # spatial_embedding is used as key and query
        # z is used as value
        attn = self.similarity(spatial_embedding)
        out = einsum('b i j, b j d -> b i d', attn, z)
        return out

        # b, n, _, h = *x.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # attn = self.attend(dots)

        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # return self.to_out(out)

class SpatialTransformer(nn.Module):
    def __init__(self, spatial_dim, z_dim, depth, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(),
                PreNorm(z_dim, FeedForward(
                    z_dim, mlp_dim, z_dim, dropout=dropout)),
                PreNorm(z_dim, FeedForward(
                    z_dim, mlp_dim, spatial_dim, dropout=dropout))
            ]))

    def forward(self, z, spatial_embedding):
        for attn, ffz, ffs in self.layers:
            z = attn(spatial_embedding, z)
            z = ffz(z) + z
            spatial_embedding = ffs(z) + spatial_embedding
        return z, spatial_embedding

class SpatialViT(nn.Module):
    def __init__(self, spatial_dim, z_dim, depth, mlp_dim,
                 dropout = 0., 
                 emb_dropout = 0.):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = SpatialTransformer(
            spatial_dim, z_dim, depth, mlp_dim, dropout)

    def forward(self, z, spatial_embedding):
        z = self.dropout(z)
        z, spatial_embedding = self.transformer(z, spatial_embedding)
        return z, spatial_embedding
