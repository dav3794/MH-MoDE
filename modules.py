from torch import nn
import torch
from transformers import PretrainedConfig
import torch.nn.functional as F
from einops import einsum, rearrange
import math
from copy import deepcopy
from time import time


class MLP(nn.Module):
    """Standard MLP module"""
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias),
        )

    def forward(self, x):
        return self.mlp(x)


class Router(nn.Module):
    """Router module for mixture models"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts

        self.expert_embeddings = nn.Linear(self.hidden_size, self.num_experts)
        torch.nn.init.xavier_uniform_(self.expert_embeddings.weight)

    def forward(self, x):      
        routing_weights = self.expert_embeddings(x)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        return routing_weights
    

class MoE(nn.Module):
    """Mixture-of-Experts feed forward"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts_per_token = config.num_experts_per_token
        self.capacity_factor = config.capacity_factor

        self.experts_inter = nn.Parameter(torch.randn(self.num_experts, self.hidden_size, self.intermediate_size))
        self.experts_out = nn.Parameter(torch.randn(self.num_experts, self.intermediate_size, self.hidden_size))
        self.expert_activation = nn.ReLU()
        self.router = Router(config)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        expert_capacity = math.ceil(batch_size * seq_len / self.num_experts * self.capacity_factor)
        routing_weights = self.router(x)

        x_flat = x.view(-1, hidden_size)
        weights_flat = routing_weights.view(-1, self.num_experts)

        # Get top-k experts for each token
        non_zero_experts = torch.topk(weights_flat, self.num_experts_per_token, dim=-1)
        topk_flat_weights = torch.zeros_like(weights_flat).scatter_(-1, non_zero_experts.indices, non_zero_experts.values)
        routing_mask = (topk_flat_weights > 0).long()

        # Assign capacity tokens to experts
        sorted_mask, sorted_indices = torch.sort(routing_mask, dim=0, descending=True)
        sorted_mask = sorted_mask[:expert_capacity]
        sorted_indices = sorted_indices[:expert_capacity]
        used_tokens_indices = sorted_indices * sorted_mask # zero indices for zero-weights tokens
        used_tokens_indices = used_tokens_indices.flatten()

        inputs_tokens = torch.index_select(x_flat, 0, used_tokens_indices) * sorted_mask.flatten().unsqueeze(-1)
        inputs_tokens = inputs_tokens.view(expert_capacity, self.num_experts, hidden_size)
        inputs_weights = torch.gather(topk_flat_weights, 0, sorted_indices)

        # Compute expert outputs
        out = einsum(inputs_tokens, self.experts_inter, 'c e h, e h i -> c e i')
        out = self.expert_activation(out)
        out = einsum(out, self.experts_out, 'c e i, e i h -> c e h')

        out = out * inputs_weights.unsqueeze(-1)
        out = out.reshape(-1, hidden_size)

        # Collect expert token embeddings back to original shape
        expert_embeddings = torch.zeros_like(x_flat).index_add_(0, used_tokens_indices, out)
        expert_embeddings = expert_embeddings.view(batch_size, seq_len, hidden_size)

        return expert_embeddings


class MoDE(nn.Module):
    """Integrated Mixture-of-Depths-and-Experts feed forward"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts # includes one no-op expert as the last expert
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts_per_token = config.num_experts_per_token
        self.capacity_factor = config.capacity_factor

        self.experts_inter = nn.Parameter(torch.randn(self.num_experts-1, self.hidden_size, self.intermediate_size))
        self.experts_out = nn.Parameter(torch.randn(self.num_experts-1, self.intermediate_size, self.hidden_size))
        self.expert_activation = nn.ReLU()
        self.router = Router(config)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        expert_capacity = math.ceil(batch_size * seq_len / self.num_experts * self.capacity_factor)
        routing_weights = self.router(x) # [batch_size, seq_len, num_experts]

        x_flat = x.view(-1, hidden_size) # [batch_size * seq_len, hidden_size]
        weights_flat = routing_weights.view(-1, self.num_experts) # [batch_size * seq_len, num_experts]

        # Get top-k experts for each token
        non_zero_experts = torch.topk(weights_flat, self.num_experts_per_token, dim=-1)
        topk_flat_weights = torch.zeros_like(weights_flat).scatter_(-1, non_zero_experts.indices, non_zero_experts.values)

        # Get no-op tokens
        noop_flat_weights = topk_flat_weights[:, -1:] # no-op expert, [batch_size*seq_len, 1]
        noop_routed = x_flat * noop_flat_weights # zeros for non-routed tokens
        
        # Assign capacity tokens to experts
        experts_flat_weights = topk_flat_weights[:, :-1] # exclude no-op expert, [batch_size*seq_len, (num_experts-1)]
        experts_routing_mask = (experts_flat_weights > 0).long()
        sorted_mask, sorted_indices = torch.sort(experts_routing_mask, dim=0, descending=True) # stable=False by default
        sorted_mask = sorted_mask[:expert_capacity] # [expert_capacity, num_experts-1]
        sorted_indices = sorted_indices[:expert_capacity] # [expert_capacity, num_experts-1]
        used_tokens_indices = sorted_indices * sorted_mask # zero indices for zero-weights tokens
        used_tokens_indices = used_tokens_indices.flatten()

        inputs_tokens = torch.index_select(x_flat, 0, used_tokens_indices) * sorted_mask.flatten().unsqueeze(-1)
        inputs_tokens = inputs_tokens.view(expert_capacity, self.num_experts-1, hidden_size)
        inputs_weights = torch.gather(experts_flat_weights, 0, sorted_indices)

        # Compute expert outputs
        out = einsum(inputs_tokens, self.experts_inter, 'c e h, e h i -> c e i')
        out = self.expert_activation(out)
        out = einsum(out, self.experts_out, 'c e i, e i h -> c e h')

        out = out * inputs_weights.unsqueeze(-1)
        out = out.reshape(-1, hidden_size)

        # Collect expert token embeddings back to original shape
        expert_embeddings = torch.zeros_like(x_flat).index_add_(0, used_tokens_indices, out) # add expert embeddings
        expert_embeddings += noop_routed
        expert_embeddings = expert_embeddings.view(batch_size, seq_len, hidden_size)

        return expert_embeddings
    

class MHMixtureWrapper(nn.Module):
    """Multi-Head wrapper for Mixture models"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.mh_input_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.mh_output_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.ReLU()

        ff_config = deepcopy(config)
        ff_config.hidden_size = self.hidden_size // self.num_heads
        self.ff = config.ff_cls(ff_config)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x = self.mh_input_layer(x)
        x = self.activation(x)

        # Split tokens into subtokens
        x = x.view(batch_size, seq_len, self.num_heads, hidden_size // self.num_heads)
        x = x.reshape(batch_size, -1, hidden_size // self.num_heads) # [batch_size, seq_len*num_heads, hidden_size//num_heads]

        # Apply FF to subtokens
        x = self.ff(x)

        # Merge subtokens back to tokens
        x = x.view(batch_size, seq_len, self.num_heads, hidden_size // self.num_heads)
        x = x.reshape(batch_size, seq_len, hidden_size)

        x = self.mh_output_layer(x)
        x = self.activation(x)
        
        return x
    

class Embedding(nn.Module):
  def __init__(self, config):
    super(Embedding, self).__init__()
    self.word_embed = nn.Embedding(config.vocab_size, config.hidden_size)
    self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, x):
    batch_size, seq_length = x.shape
    device = x.device
    positions = torch.arange(0, seq_length).expand(
        batch_size, seq_length).to(device)
    embedding = self.word_embed(x) + self.pos_embed(positions)
    return self.dropout(embedding)


class MHSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super(MHSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)

    def forward(self, embeddings):
        batch_size, seq_length, hidden_size = embeddings.size()

        result = self.qkv(embeddings)
        q, k, v = rearrange(result, 'b s (qkv nah hdsz) -> qkv b nah s hdsz', nah=self.num_attention_heads, qkv=3).unbind(0)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(hidden_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        contextualized_layer = torch.matmul(attention_probs, v)

        outputs = rearrange(contextualized_layer, 'b nah s hdsz -> b s (nah hdsz)')
        return outputs
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MHSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        if config.mh_moe:
            self.intermediate = MHMixtureWrapper(config)
        else:
            self.intermediate = config.ff_cls(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x =  x + self.norm1(self.dropout(self.attention(x)))
        x =  x + self.norm2(self.dropout(self.intermediate(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embedding(config)
        self.layer = nn.Sequential(*[TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, labels=None):
        embedding_output = self.embeddings(input_ids)
        encoding = self.layer(embedding_output)
        pooled_encoding = encoding.mean(dim=1)
        logits = self.classifier(pooled_encoding)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {
            'loss': loss,
            'logits': logits,
        }
