import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hyperparameters describing a GPT-2 model architecture."""

    block_size: int = 1024  # max sequence length
    vocab_size: int = 50_257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a lower-triangular mask."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the attention projections and causal mask.

        Args:
            config: Model configuration providing the embedding and head sizes.
        """
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # key, query value projections for all heads, but in a batch
        self.c_attn: nn.Linear = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.c_proj: nn.Linear = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.n_head: int = config.n_head
        self.n_embd: int = config.n_embd

        # more of a "mask" instead of a bias; following the OpenAI/HF naming
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention to ``x``.

        Args:
            x: Input activations of shape ``(B, T, C)``.

        Returns:
            Output activations of shape ``(B, T, C)``.
        """
        # batch size, sequence length, embedding dimensionality (n_embd)
        B: int
        T: int
        C: int
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv: torch.Tensor = self.c_attn(x)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y: torch.Tensor = att @ v

        # re-assemble all head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network used inside each transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the two linear layers and the GELU activation.

        Args:
            config: Model configuration providing the embedding dimension.
        """
        super().__init__()

        self.c_fc: nn.Linear = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu: nn.GELU = nn.GELU(approximate='tanh')
        self.c_proj: nn.Linear = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to ``x``.

        Args:
            x: Input activations of shape ``(..., n_embd)``.

        Returns:
            Output activations with the same shape as ``x``.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block with attention and MLP residual paths."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize the layer norms, attention, and MLP submodules.

        Args:
            config: Model configuration for the transformer block.
        """
        super().__init__()

        self.ln_1: nn.LayerNorm = nn.LayerNorm(config.n_embd)
        self.attn: CausalSelfAttention = CausalSelfAttention(config)
        self.ln_2: nn.LayerNorm = nn.LayerNorm(config.n_embd)
        self.mlp: MLP = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention and MLP sublayers with residual connections.

        Args:
            x: Input activations of shape ``(B, T, n_embd)``.

        Returns:
            Output activations of shape ``(B, T, n_embd)``.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """A decoder-only GPT-2 style language model."""

    def __init__(self, config: GPTConfig) -> None:
        """Build the token/position embeddings, transformer blocks, and LM head.

        Args:
            config: Model configuration describing the architecture.
        """
        super().__init__()

        self.config: GPTConfig = config
        self.transformer: nn.ModuleDict = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head: nn.Linear = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for a batch of token indices.

        Args:
            idx: Token indices of shape ``(B, T)``.

        Returns:
            Logits of shape ``(B, T, vocab_size)``.

        Raises:
            AssertionError: If ``T`` exceeds the configured block size.
        """
        # idx is of shape (B, T)
        B: int
        T: int
        B, T = idx.size()
        assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'

        # forward the token and position embeddings
        pos: torch.Tensor = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb: torch.Tensor = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb: torch.Tensor = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x: torch.Tensor = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits: torch.Tensor = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        """Load a pretrained GPT-2 checkpoint from Hugging Face.

        Args:
            model_type: One of ``"gpt2"``, ``"gpt2-medium"``,
                ``"gpt2-large"``, or ``"gpt2-xl"``.

        Returns:
            A ``GPT`` instance populated with the checkpoint weights.

        Raises:
            AssertionError: If ``model_type`` is unsupported or the parameter
                names/shapes do not match between implementations.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        # print('Loading weights from pretrained gpt: %s' % model_type)

        # n_layer, n_head, and n_embd are determined from model_type
        config_args: dict[str, int] = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50_257  # always 50,257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config: GPTConfig = GPTConfig(**config_args)
        model: GPT = GPT(config)
        sd: dict[str, torch.Tensor] = model.state_dict()
        sd_keys: list[str] = list(sd.keys())
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask/buffer

        # init a huggingface/transformers model
        model_hf: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf: dict[str, torch.Tensor] = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and math in names and shapes
        sd_keys_hf: list[str] = list(sd_hf.keys())
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same here, just the mask (buffer)

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed: list[str] = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
