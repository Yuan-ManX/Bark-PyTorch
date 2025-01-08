import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    """
    带可选偏置的层归一化（LayerNorm）模块。

    PyTorch 的 `F.layer_norm` 函数允许显式地指定是否使用偏置（bias），而 `nn.LayerNorm` 默认包含偏置。
    该类提供了可选偏置的层归一化实现。

    参数:
        ndim (int): 输入张量的维度，用于层归一化。
        bias (bool): 是否使用偏置，默认为 True。
    """

    def __init__(self, ndim, bias):
        super().__init__()
        # 可学习的缩放参数（gamma）
        self.weight = nn.Parameter(torch.ones(ndim))
        # 可学习的偏置参数（beta），如果 bias 为 False，则为 None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """
        前向传播方法，执行层归一化。

        参数:
            input (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 层归一化后的张量。
        """
        # 使用 PyTorch 的层归一化函数，eps 为 1e-5
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    """
    多层感知机（MLP）模块。

    该模块实现了 Transformer 模型中的前馈神经网络部分，通常用于处理和转换输入特征。
    包含两个线性变换层和一个 GELU 激活函数，以及一个 Dropout 层。

    参数:
        config: 配置参数，包含模型的各种配置参数。
    """

    def __init__(self, config):
        super().__init__()
        # 第一个线性变换层，输入维度为 n_embd，输出维度为 4 * n_embd
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # 第二个线性变换层，输入维度为 4 * n_embd，输出维度为 n_embd
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout 层，dropout 概率为 config.dropout
        self.dropout = nn.Dropout(config.dropout)
        # GELU 激活函数
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        前向传播方法，执行 MLP 的计算。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: MLP 处理后的输出张量。
        """
        # 第一个线性变换
        x = self.c_fc(x)
        # 应用 GELU 激活函数
        x = self.gelu(x)
        # 第二个线性变换
        x = self.c_proj(x)
        # 应用 Dropout
        x = self.dropout(x)
        # 返回 MLP 的输出
        return x


class Block(nn.Module):
    """
    Transformer 块（Block）模块。

    该模块实现了 Transformer 模型中的一个基本块，包括层归一化、因果自注意力机制和 MLP。
    每个块执行以下操作：
        1. 对输入进行层归一化。
        2. 应用因果自注意力机制。
        3. 将注意力输出与原始输入进行残差连接。
        4. 对结果进行层归一化。
        5. 应用 MLP。
        6. 将 MLP 输出与步骤 3 的结果进行残差连接。

    参数:
        config: 配置参数，包含模型的各种配置参数。
        layer_idx (int): 当前块的索引，用于标识层的位置。
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        # 第一个层归一化层
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 因果自注意力机制
        self.attn = CausalSelfAttention(config)
        # 第二个层归一化层
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        # 当前块的索引
        self.layer_idx = layer_idx

    def forward(self, x, past_kv=None, use_cache=False):
        """
        前向传播方法，执行 Transformer 块的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, Seq_Len, Dim)。
            past_kv (Optional[torch.Tensor]): 过去的键和值张量，用于缓存，默认为 None。
            use_cache (bool): 是否使用缓存，默认为 False。

        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 块处理后的输出张量和更新后的缓存。
        """
        # 应用第一个层归一化，并执行因果自注意力机制
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        # 将注意力输出与原始输入进行残差连接
        x = x + attn_output
        # 应用第二个层归一化，并执行 MLP
        x = x + self.mlp(self.ln_2(x))
        # 返回块处理后的输出和更新后的缓存
        return (x, prev_kvs)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制模块。

    该模块实现了因果自注意力机制，用于捕捉序列中的长距离依赖关系，同时确保在预测当前时间步时只能看到过去的信息。
    支持使用 Flash Attention（如果可用）以加速计算。

    参数:
        config: 配置参数，包含模型的各种配置参数。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # 对所有头进行键（key）、查询（query）和值（value）的线性投影，但作为一个批次处理
        # 线性变换层，输出维度为 3 * n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        # 输出线性投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        # 正则化层
        self.attn_dropout = nn.Dropout(config.dropout) # 注意力 Dropout
        self.resid_dropout = nn.Dropout(config.dropout) # 残差 Dropout
        self.n_head = config.n_head # 注意力头的数量
        self.n_embd = config.n_embd # 嵌入维度
        self.dropout = config.dropout # Dropout 概率
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # 创建因果掩码，以确保注意力仅应用于输入序列的左侧
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size)) # 创建下三角掩码

    def forward(self, x, past_kv=None, use_cache=False):
        """
        前向传播方法，执行因果自注意力机制。

        参数:
            x (torch.Tensor): 输入张量，形状为 (B, T, C)。
            past_kv (Optional[torch.Tensor]): 过去的键和值张量，用于缓存，默认为 None。
            use_cache (bool): 是否使用缓存，默认为 False。

        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 注意力输出和更新后的缓存。
        """
        # 获取批次大小 (B)、序列长度 (T) 和嵌入维度 (C)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 对输入张量进行线性变换，得到查询 (q)、键 (k) 和值 (v)，并移动头维度到批次维度
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # 获取键张量的总长度
        FULL_T = k.shape[-2]

        if use_cache is True:
            # 如果使用缓存，则返回当前的键和值作为缓存
            present = (k, v)
        else:
            present = None

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 因果自注意力机制
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # 使用 Flash Attention 进行高效计算
            if past_kv is not None:
                # When `past_kv` is provided, we're doing incremental decoding and `q.shape[2] == 1`: q only contains
                # the query for the last token. scaled_dot_product_attention interprets this as the first token in the
                # sequence, so if is_causal=True it will mask out all attention from it. This is not what we want, so 
                # to work around this we set is_causal=False.
                # 当提供 past_kv 时，进行增量解码，q.shape[2] == 1：q 仅包含最后一个 token 的查询
                # scaled_dot_product_attention 将其解释为序列中的第一个 token，因此如果 is_causal=True，它将屏蔽所有来自它的注意力。
                # 这不是我们想要的，所以为了解决这个问题，我们将 is_causal 设置为 False。
                is_causal = False
            else:
                is_causal = True

            # 使用 Flash Attention 计算注意力
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal)
        else:
            # manual implementation of attention
            # 手动实现注意力机制
            # 计算注意力分数
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 应用因果掩码
            att = att.masked_fill(self.bias[:,:,FULL_T-T:FULL_T,:FULL_T] == 0, float('-inf'))
            # 对注意力分数进行 softmax 归一化
            att = F.softmax(att, dim=-1)
            # 应用注意力 Dropout
            att = self.attn_dropout(att)
            # 计算加权的值
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 重塑张量形状以合并多头输出
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # 输出线性投影
        y = self.resid_dropout(self.c_proj(y)) # 应用残差 Dropout 和输出线性变换
        # 返回注意力输出和缓存
        return (y, present)


@dataclass
class GPTConfig:
    """
    GPT 模型配置类。

    该类定义了 GPT 模型的各种配置参数，用于控制模型的结构和行为。
    """
    block_size: int = 1024  # 上下文窗口大小，即模型一次可以处理的最大序列长度，默认为 1024
    input_vocab_size: int = 10_048  # 输入词汇表大小，默认为 10,048
    output_vocab_size: int = 10_048  # 输出词汇表大小，默认为 10,048
    n_layer: int = 12  # Transformer 层的数量，默认为 12
    n_head: int = 12  # 每个注意力头的数量，默认为 12
    n_embd: int = 768  # 嵌入维度，默认为 768
    dropout: float = 0.0  # Dropout 概率，默认为 0.0
    bias: bool = True  # 是否在线性层和层归一化中使用偏置，默认为 True（类似于 GPT-2）。设置为 False 时，模型性能略好且速度更快


class GPT(nn.Module):
    """
    GPT 模型类。

    该类实现了 GPT 模型，包括词嵌入、位置嵌入、多个 Transformer 层、层归一化以及语言模型头（LM Head）。
    支持增量解码（incremental decoding）和缓存机制，以提高推理效率。
    """

    def __init__(self, config):
        """
        初始化 GPT 模型。

        参数:
            config: GPT 模型配置参数，包含模型的各种配置参数。
        """
        super().__init__()
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        # 保存配置参数
        self.config = config

        # 定义 Transformer 模块，包括词嵌入、位置嵌入、Dropout、多个 Transformer 块以及层归一化
        self.transformer = nn.ModuleDict(dict(
            # 词嵌入层，输入词汇表大小为 input_vocab_size，嵌入维度为 n_embd
            wte = nn.Embedding(config.input_vocab_size, config.n_embd),
            # 位置嵌入层，块大小为 block_size，嵌入维度为 n_embd
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout 层，dropout 概率为 dropout
            drop = nn.Dropout(config.dropout),
            # 多个 Transformer 块，列表长度为 n_layer
            h = nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
            # 最终层归一化层
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # 语言模型头（LM Head），将嵌入维度映射到输出词汇表大小，不使用偏置
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

    def get_num_params(self, non_embedding=True):
        """
        计算模型参数的数量。

        参数:
            non_embedding (bool, 可选): 是否排除嵌入参数，默认为 True。

        返回:
            int: 模型参数的数量。
        """
        # 计算所有参数的数量
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 排除词嵌入参数
            n_params -= self.transformer.wte.weight.numel()
            # 排除位置嵌入参数
            n_params -= self.transformer.wpe.weight.numel()
        # 返回参数数量
        return n_params

    def forward(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        """
        前向传播方法，执行 GPT 模型的前向传播。

        参数:
            idx (torch.Tensor): 输入 token 索引张量，形状为 (B, T)。
            merge_context (bool, 可选): 是否合并上下文，默认为 False。
            past_kv (Optional[Tuple[torch.Tensor, torch.Tensor]], 可选): 过去的键和值张量，用于缓存，默认为 None。
            position_ids (Optional[torch.Tensor], 可选): 位置 ID 张量，可选。
            use_cache (bool, 可选): 是否使用缓存，默认为 False。

        返回:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 语言模型输出 logits 和更新后的缓存。
        """
        device = idx.device
        # 获取批次大小 (B) 和序列长度 (T)
        b, t = idx.size()
        if past_kv is not None:
            assert t == 1
            # 获取词嵌入，形状为 (B, T, n_embd)
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        else:
            if merge_context:
                assert(idx.shape[1] >= 256+256+1)
                # 计算新的序列长度
                t = idx.shape[1] - 256
            else:
                assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            # forward the GPT model itself
            # 前向传播 GPT 模型本身
            if merge_context:
                # 合并上下文：将前 256 个 token 和接下来的 256 个 token 进行合并，然后添加最后一个 token
                tok_emb = torch.cat([
                    self.transformer.wte(idx[:,:256]) + self.transformer.wte(idx[:,256:256+256]),
                    self.transformer.wte(idx[:,256+256:])
                ], dim=1)
            else:
                # 获取词嵌入，形状为 (B, T, n_embd)
                tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if past_kv is None:
            # 如果没有过去的键和值，则过去的序列长度为 0
            past_length = 0
            # 初始化 past_kv 为 None 列表
            past_kv = tuple([None] * len(self.transformer.h))
        else:
            # 获取过去的序列长度
            past_length = past_kv[0][0].size(-2)

        if position_ids is None:
            # 生成位置 ID
            position_ids = torch.arange(past_length, t + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # 形状为 (1, T)
            assert position_ids.shape == (1, t)

        # 获取位置嵌入，形状为 (1, T, n_embd)
        pos_emb = self.transformer.wpe(position_ids) # position embeddings of shape (1, t, n_embd)

        # 应用 Dropout 并添加位置嵌入
        x = self.transformer.drop(tok_emb + pos_emb)

        # 如果使用缓存，则初始化 new_kv，否则为 None
        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.transformer.h, past_kv)):
            # 应用 Transformer 块
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            if use_cache:
                # 更新缓存
                new_kv = new_kv + (kv,)

        # 应用最终层归一化
        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        # 推理时的微型优化：仅在最后一个位置应用 lm_head
        # 仅在最后一个时间步计算 logits，形状为 (B, 1, output_vocab_size)
        logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim

        # 返回 logits 和更新后的缓存
        return (logits, new_kv)
