from einops import rearrange, reduce
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import os

if 'ENTROPY_LOSS' in os.environ:
    print("ENTROPY_LOSS is set")
    ENTROPY_LOSS = True
else:
    ENTROPY_LOSS = False

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)

def linfnorm(t, dim = -1, eps = 1e-6):
    # L-infinity normalization: project features onto the surface of the unit
    # hypercube so that max_i |t_i| == 1 (||t||_inf == 1). This is the proximal
    # representation used in ViQ stage 2-1 to keep features close to the
    # quantization anchors before discretization.
    return t / t.abs().amax(dim=dim, keepdim=True).clamp_min(eps)

def compute_entropy_loss(
    logits,
    temperature=0.01,
    sample_minimization_weight=1.0,
    batch_maximization_weight=1.0,
    eps=1e-5,
):
    """
    Entropy loss of unnormalized logits

    logits: Affinities are over the last dimension

    https://github.com/google-research/magvit/blob/05e8cfd6559c47955793d70602d62a2f9b0bdef5/videogvt/train_lib/losses.py#L279
    LANGUAGE MODEL BEATS DIFFUSION — TOKENIZER IS KEY TO VISUAL GENERATION (2024)
    """
    probs = F.softmax(logits / temperature, -1)
    log_probs = F.log_softmax(logits / temperature + eps, -1)

    avg_probs = reduce(probs, "... D -> D", "mean")

    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + eps))

    sample_entropy = -torch.sum(probs * log_probs, -1)
    sample_entropy = torch.mean(sample_entropy)

    loss = (sample_minimization_weight * sample_entropy) - (
        batch_maximization_weight * avg_entropy
    )

    return sample_entropy, avg_entropy, loss

class IBQ(nn.Module):
    def __init__(
            self, 
            dim,
            codebook_size,
            codebook_dim,
            # n_embed, 
            # embed_dim, 
            l2_norm=True, 
            beta=1.0, 
            quantization_temp=1.0, 
            input_format='blc',
            limit = 'none',
            symmetry_vq=False,
            ts_factor=1,
            skip_quant_prob=0.1
        ):
        super().__init__()

        self.n_embed = codebook_size
        self.embed_dim = codebook_dim
        self.dim = dim
        self.skip_quantization_prob = skip_quant_prob
        self.use_entropy_loss = ENTROPY_LOSS

        has_projections = (dim != codebook_dim)
        self.project_in = nn.Linear(dim, codebook_dim) if has_projections else nn.Identity()
        if symmetry_vq:
            self.project_out = nn.Linear(codebook_dim, dim) if has_projections else nn.Identity()
        else:
            self.project_out = None

        self.forward_times = 0
        self.analysis_code_collection = []
        self.history_code_collection = []

        self.l2_norm = l2_norm
        self.beta = beta
        assert input_format in ['bchw', 'blc']
        self.input_format = input_format
        self.quantization_temp = quantization_temp

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1 / self.n_embed, 1 / self.n_embed)
        self.bits_per_index = int(np.ceil(np.log2(self.n_embed)))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)
        
        self.limit = limit

        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)

    def forward(self, z, slen=None, image_sizes=None):
        # we always use 1 N C feature 
        if self.input_format == 'bchw':
            z = rearrange(z, 'b c h w -> b h w c')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {z.shape[-1]}'
        z = self.project_in(z)

        if self.l2_norm:
            # we will normalize the input and embedding
            z = l2norm(z)
            z_flatten = z.reshape(-1, self.embed_dim) # N c
            embedding_weight = l2norm(self.embedding.weight) # d c
        else:
            z_flatten = z.reshape(-1, self.embed_dim)
            embedding_weight = self.embedding.weight # d c
        
        d = torch.cdist(z_flatten, embedding_weight)


        if self.training:
            logits = -d / self.quantization_temp # N d
            soft_one_hot = F.softmax(logits, dim=1)
            min_encoding_indices = soft_one_hot.max(1, keepdim=True)[1]
            hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, min_encoding_indices, 1.0)
            one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

            z_q = torch.einsum('b n, n d -> b d', one_hot, self.embedding.weight).view(z.shape)
            z_q_2 = torch.einsum('b n, n d -> b d', hard_one_hot, self.embedding.weight).view(z.shape)

            if self.use_entropy_loss:
                sample_entropy, avg_entropy, entropy_loss= compute_entropy_loss(
                    logits=logits, 
                )

            if slen is not None:
                zs = z.split(slen, dim=1)
                quantizeds = z_q.split(slen, dim=1)
                quantizeds_2 = z_q_2.split(slen, dim=1)
                commit_loss_list = []
                for _z, _q, _q_2 in zip(zs, quantizeds, quantizeds_2):
                    _commit_loss = torch.mean((_q - _z) ** 2) + torch.mean((_q_2.detach() - _z) ** 2) + self.beta * \
                        torch.mean((_q_2 - _z.detach()) ** 2)
                    if self.use_entropy_loss:
                        _commit_loss += entropy_loss
                    commit_loss_list.append(_commit_loss)
                commit_loss = commit_loss_list
            else:
                commit_loss = torch.mean((z_q - z) ** 2) + torch.mean((z_q_2.detach() - z) ** 2) + self.beta * \
                            torch.mean((z_q_2 - z.detach()) ** 2)
                if self.use_entropy_loss:
                    commit_loss += entropy_loss

        else:
            min_encoding_indices = torch.argmin(d, dim=1) # N d 
            z_q = self.embedding(min_encoding_indices).view(z.shape)
            if slen is not None:
                commit_loss = [self.zero] * len(slen)
            else:
                commit_loss = self.zero

        # analysis the usage precent.
        global_rank = torch.distributed.get_rank()
        if global_rank == 0:
            tensor_in = torch.unique(min_encoding_indices.flatten())
            self.analysis_code_collection.append(tensor_in)
            self.forward_times += 1
            freq = 20 if slen is not None else 20*64

            if self.forward_times % freq == 0:
                used_code = torch.unique(torch.cat(self.analysis_code_collection, dim=0))
                num_unique = used_code.numel()
                self.analysis_code_collection = []
                print(f'\n\nRank {global_rank} - Round{self.forward_times}: {freq} times forward Usage is {num_unique}/{self.n_embed} = {num_unique / self.n_embed:.2%}')
                
                self.history_code_collection.append(used_code)
                if len(self.history_code_collection) > 20:
                    self.history_code_collectio = self.history_code_collection[-20:]
                used_code = torch.unique(torch.cat(self.history_code_collection, dim=0))
                num_unique = used_code.numel()
                print(f'Rank {global_rank} - Round{self.forward_times}: History {20 * freq } Usage is {num_unique}/{self.n_embed} = {num_unique / self.n_embed:.2%} \n\n ')

        if self.training and self.skip_quantization_prob > 0.0:
            # TODO: here we shold mask it on seqlen level...
            z_q = torch.where(
                torch.rand_like(z_q[:, 0:1, 0:1]).expand_as(z_q) <= self.skip_quantization_prob,
                z, z_q,
            )

        if self.project_out is not None:
            z_q = self.project_out(z_q)
            assert z_q.size()[-1] == self.dim

        if self.input_format == 'bchw':
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        
        # to align with fake quantizer
        if self.limit == 'none':
            z_q = z_q
        elif self.limit == 'l2':
            z_q = l2norm(z_q)
        elif self.limit == 'l_infinite':
            z_q = linfnorm(z_q)
        elif self.limit == 'tanh':
            tanh = nn.Tanh()
            z_q = tanh(z_q)
        else:
            raise ValueError(f'Unknown limit type: {self.limit}')

        min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[1])

        
        return z_q, commit_loss, {'indices': min_encoding_indices}

    def get_codebook_entry(self, indices):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        # to align with fake quantizer
        if self.limit == 'none':
            z_q = z_q
        elif self.limit == 'l2':
            z_q = l2norm(z_q)
        elif self.limit == 'l_infinite':
            z_q = linfnorm(z_q)
        elif self.limit == 'tanh':
            tanh = nn.Tanh()
            z_q = tanh(z_q)
        else:
            raise ValueError(f'Unknown limit type: {self.limit}')

        return z_q


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, l2_norm, beta, input_format='bchw'):
        super().__init__()

        self.n_embed = n_embed
        self.dim = self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        assert input_format in ['bchw', 'blc']
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed)
        self.bits_per_index = int(np.ceil(np.log2(n_embed)))

    def forward(self, z):
        batch = z.shape[0]
        if self.input_format == 'bchw':
            z = rearrange(z, 'b c h w -> b h w c')

        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_flatten = z.reshape(-1, self.embed_dim)
            embedding_weight = F.normalize(self.embedding.weight, dim=-1)
            d = -z_flatten @ embedding_weight.t()
        else:
            z_flatten = z.reshape(-1, self.embed_dim)
            d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * z_flatten @ self.embedding.weight.t()

        min_encoding_indices = torch.argmin(d.detach(), dim=1)
        if not self.training:
            used_codes = torch.unique(min_encoding_indices, return_counts=False)
        else:
            used_codes = None
        cb_usage = F.one_hot(min_encoding_indices, self.n_embed).sum(0)
        cb_entropy = self.get_entropy(cb_usage)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        # fix the issue with loss scaling
        # loss weight should not associate with the dimensionality of words
        loss = self.beta * torch.mean(((z_q.detach() - z) ** 2).sum(dim=-1)) + torch.mean(((z_q - z.detach()) ** 2).sum(dim=-1))

        z_q = z + (z_q - z).detach()
        if self.input_format == 'bchw':
            z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, {"H":cb_entropy, "used_codes": used_codes, 'indices': min_encoding_indices.view(batch, -1)}

    def get_entropy(self, count, eps=1e-4):
        probs = (count + eps) / (count + eps).sum()
        H = -(probs * torch.log(probs)).sum()
        return H


    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        if self.input_format == 'bchw':
            h = w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, l2_norm, beta, decay=0.99, eps=1e-5, random_restart=True, restart_threshold=1.0, input_format='bchw'):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.random_restart = random_restart
        self.restart_threshold = restart_threshold
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed) # TODO (yzhao): test other initialization methods 
        self.register_buffer("ema_cluster_size", torch.zeros(self.n_embed))
        self.embedding_avg = nn.Parameter(torch.Tensor(self.n_embed, self.embed_dim))
        self.embedding_avg.data.copy_(self.embedding.weight.data)

    def _tile(self, z):
        n_z, embedding_dim = z.shape
        if n_z < self.n_embed:
            n_repeats = (self.n_embed + n_z - 1) // n_z
            std = 0.01 / np.sqrt(embedding_dim)
            z = z.repeat(n_repeats, 1)
            z = z + torch.randn_like(z) * std
        return z

    def forward(self, z):
        if self.input_format == 'bchw':
            z = rearrange(z, 'b c h w -> b h w c')
        z_flatten = z.reshape(-1, self.embed_dim)

        d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * z_flatten @ self.embedding.weight.t()

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.n_embed, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_q = F.normalize(z_q, dim=-1)

        if self.training:
            # EMA update cluster size
            encodings_sum = encodings.sum(0)
            if dist.is_initialized(): dist.all_reduce(encodings_sum)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1-self.decay)

            # EMA update of the embedding vectors
            dw = encodings.t() @ z_flatten
            if dist.is_initialized(): dist.all_reduce(dw)
            self.embedding_avg.data.mul_(self.decay).add_(dw, alpha=1-self.decay)
 
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size)
            weights = (self.ema_cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            self.embedding.weight.data = self.embedding_avg.data / weights.unsqueeze(1)

            if self.random_restart:
                zz = self._tile(z_flatten)
                _k_rand = zz[torch.randperm(zz.size(0))][:self.n_embed]
                if dist.is_initialized(): dist.broadcast(_k_rand, 0)
                usage = (self.ema_cluster_size.view(-1, 1) > self.restart_threshold).float()
                self.embedding.weight.data.mul_(usage).add_(_k_rand * (1 - usage))

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        z_q = z + (z_q - z).detach()
        if self.input_format == 'bchw':
            z_q = rearrange(z_q, 'b h w c -> b c h w')
        # TODO (yzhao): monitor utility of the dictionary
        return z_q, loss, {}