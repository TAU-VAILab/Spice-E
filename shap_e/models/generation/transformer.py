import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import copy
import torch.nn as nn

from shap_e.models.nn.checkpoint import checkpoint

from .pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType
from .util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, x_cond=None):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x
    

class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        no_one_conv: bool = False,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.no_one_conv = no_one_conv
        self.c_kv = nn.Linear(width, width * 2, device=device, dtype=dtype)
        self.c_q1 = nn.Linear(width, width, device=device, dtype=dtype)
        self.c_q2 = nn.Linear(width, width, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        self.zero_conv = nn.Conv1d(in_channels=n_ctx, 
                                    out_channels=n_ctx, 
                                    kernel_size=1,
                                    device=device,
                                    dtype=dtype)
        self.one_conv = nn.Conv1d(in_channels=n_ctx,
                                    out_channels=n_ctx,
                                    kernel_size=1,
                                    device=device,
                                    dtype=dtype)
        with torch.no_grad():
            self.zero_conv.weight.zero_()
            self.zero_conv.bias.zero_()
            self.one_conv.weight.copy_(torch.eye(n_ctx).unsqueeze(dim=-1))
            self.one_conv.bias.zero_()
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_q1, init_scale)
        init_linear(self.c_q2, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, x_cond=None):
        kv = self.c_kv(x)
        q1 = self.c_q1(x)
        if x_cond != None:
            q2 = self.c_q2(x_cond)
        else:
            q2 = self.c_q2(x)
        if self.no_one_conv:
            q = q1 + self.zero_conv(q2)
        else:
            q = self.one_conv(q1) + self.zero_conv(q2)
        x = torch.cat([q, kv], dim=-1)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x

    def get_weights_from_regular_attention(self, MultiheadAttention: nn.Module):
        with torch.no_grad():
            self.c_kv.weight.copy_(MultiheadAttention.c_qkv.weight[self.width:, :])
            self.c_kv.bias.copy_(MultiheadAttention.c_qkv.bias[self.width:])
            self.c_q1.weight.copy_(MultiheadAttention.c_qkv.weight[:self.width, :])
            self.c_q1.bias.copy_(MultiheadAttention.c_qkv.bias[:self.width])
            self.c_q2.weight.copy_(MultiheadAttention.c_qkv.weight[:self.width, :])
            self.c_q2.bias.copy_(MultiheadAttention.c_qkv.bias[:self.width])
            self.c_proj.weight.copy_(MultiheadAttention.c_proj.weight)


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.conditional = False
        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)
    
    def make_conditional(self):
        self.conditional = True
        self.ln1_cond = copy.deepcopy(self.ln_1)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor=None):
        if self.conditional:
            x = x + self.attn(self.ln_1(x), self.ln1_cond(x_cond))
        else:
            x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


# TODO (ES): Might want to refactor this later
class TransformerControl(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        n_ctrl_layers: int = 0,
        init_scale: float = 0.25,
    ):
        super().__init__()
        assert n_ctrl_layers <= layers, \
            f"n_ctrl_layers ({n_ctrl_layers}) must be <= layers ({layers})"
        self.cross_mode = False
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.width = width
        self.layers = layers
        self.heads = heads
        self.device = device
        self.init_scale = init_scale
        self.ctrl_layers = n_ctrl_layers
        self.reg_layers = layers - n_ctrl_layers
        self.i_start = self.reg_layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(self.layers)
            ]
        )

    def make_ca_layers(self, conditional: bool=False, no_one_conv: bool=False):
        self.cross_mode = True
        self.conditional = conditional

        # make cross attention control resblocks
        self.ca_ctrl_blocks = nn.ModuleList([])
        self.i_start = self.reg_layers
        if self.reverse:
            self.i_start = 0
        for i in range(self.ctrl_layers):
            resblock_copy = copy.deepcopy(self.resblocks[i + self.i_start])
            ca_block = MultiheadCrossAttention(
                    device=self.device,
                    dtype=self.dtype,
                    n_ctx=self.n_ctx,
                    width=self.width,
                    heads=self.heads,
                    init_scale=self.init_scale,
                    no_one_conv=no_one_conv,
                )
            ca_block.get_weights_from_regular_attention(resblock_copy.attn)
            if self.conditional:
                resblock_copy.make_conditional()
            resblock_copy.attn = ca_block
            self.ca_ctrl_blocks.append(resblock_copy)

    def make_ctrl_layers(self, 
                         num_ctrl_layers: int=24, 
                         reverse: bool=False, 
                         cross_mode=True,
                         conditional=True,
                         no_one_conv=True):
        self.reverse = reverse
        self.ctrl_layers = num_ctrl_layers
        self.reg_layers = self.layers - self.ctrl_layers
        assert self.reg_layers >= 0, f"reg_layers must be >= 0, got {self.reg_layers}"
        if cross_mode:
            return self.make_ca_layers(conditional=conditional, no_one_conv=no_one_conv)
        self.trainable_copies = nn.ModuleList([])
        self.zero_convs = nn.ModuleList([])

        # make control resblocks
        self.i_start = self.reg_layers
        if self.reverse:
            self.i_start = 0
        for i in range(self.ctrl_layers):
            trainable_copy = copy.deepcopy(self.resblocks[i + self.i_start])
            zero_conv = nn.Conv1d(in_channels=self.n_ctx, 
                                    out_channels=self.n_ctx, 
                                    kernel_size=1,
                                    device=self.device,
                                    dtype=self.dtype)
            with torch.no_grad():
                zero_conv.weight.zero_()
                zero_conv.bias.zero_()
            self.trainable_copies.append(trainable_copy)
            self.zero_convs.append(zero_conv)

        # make zero conv for condition
        self.zero_conv_cond = nn.Conv1d(in_channels=self.n_ctx, 
                                    out_channels=self.n_ctx, 
                                    kernel_size=1,
                                    device=self.device,
                                    dtype=self.dtype)
        with torch.no_grad():
            self.zero_conv_cond.weight.zero_()
            self.zero_conv_cond.bias.zero_()
    
    def set_controlnet_full_backbone(self, verbose=False):
        if verbose:
            print(f"Setting up parameters in full backbone mode...")
            print(f"self.reverse - {self.reverse}")
        for i, block in enumerate(self.resblocks):
            if verbose:
                print(f"block {i}")
            if ((i < self.reg_layers) and not self.reverse) or ((i >= self.ctrl_layers) and self.reverse):
                if verbose:    
                    print(f"param should be trainable - {i}")
                for _, param in block.named_parameters():
                    param.requires_grad = True
        
    def forward(self, x: torch.Tensor, **kwargs):
        for i, block in enumerate(self.resblocks):
            # if on first control block
            if i == self.i_start:
                if self.cross_mode:
                    if 'cond_prepared' in kwargs.keys():
                        x = self.ca_ctrl_blocks[0](x, kwargs['cond_prepared'])
                    else:
                        x = self.ca_ctrl_blocks[0](x)
                else:
                    x_cond = x
                    if 'cond_prepared' in kwargs.keys():
                        x_cond = kwargs['cond_prepared']
                        x_cond = x + self.zero_conv_cond(x_cond)
                    x_ctrl = self.trainable_copies[0](x_cond)
                    x = block(x)
                    x = x + self.zero_convs[0](x_ctrl)

            # if on other control blocks
            elif ((i > self.reg_layers) and not self.reverse) or ((i < self.ctrl_layers) and self.reverse):
                if self.cross_mode:
                    if 'cond_prepared' in kwargs.keys():
                        x = self.ca_ctrl_blocks[i - self.reg_layers](x, kwargs['cond_prepared'])
                    else:
                        x = self.ca_ctrl_blocks[i - self.reg_layers](x)
                else:
                    x_ctrl = self.trainable_copies[i - self.reg_layers](x_ctrl)
                    x = block(x)
                    x = x + self.zero_convs[i - self.reg_layers](x_ctrl)

            # if on regular blocks
            else:
                x = block(x)
        return x


class TransformerReverseControl(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        n_ctrl_layers: int = 12,
        init_scale: float = 0.25,
    ):
        super().__init__()
        assert n_ctrl_layers <= layers, \
            f"n_ctrl_layers ({n_ctrl_layers}) must be <= layers ({layers})"
        self.n_ctx = n_ctx
        self.dtype = dtype
        self.width = width
        self.layers = layers
        self.heads = heads
        self.device = device
        self.init_scale = init_scale
        self.ctrl_layers = n_ctrl_layers
        self.reg_layers = layers - n_ctrl_layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(self.layers)
            ]
        )

    def make_ctrl_layers(self, num_ctrl_layers: int=13):
        self.ctrl_layers = num_ctrl_layers
        self.reg_layers = self.layers - self.ctrl_layers
        self.trainable_copies = nn.ModuleList([])
        self.zero_convs = nn.ModuleList([])

        # make control resblocks
        for i in range(self.ctrl_layers):
            trainable_copy = copy.deepcopy(self.resblocks[i])
            zero_conv = nn.Conv1d(in_channels=self.n_ctx, 
                                    out_channels=self.n_ctx, 
                                    kernel_size=1,
                                    device=self.device,
                                    dtype=self.dtype)
            with torch.no_grad():
                zero_conv.weight.zero_()
                zero_conv.bias.zero_()
            self.trainable_copies.append(trainable_copy)
            self.zero_convs.append(zero_conv)

        # make zero conv for condition
        self.zero_conv_cond = nn.Conv1d(in_channels=self.n_ctx, 
                                    out_channels=self.n_ctx, 
                                    kernel_size=1,
                                    device=self.device,
                                    dtype=self.dtype)
        with torch.no_grad():
            self.zero_conv_cond.weight.zero_()
            self.zero_conv_cond.bias.zero_()
        

    def forward(self, x: torch.Tensor, **kwargs):
        for i, block in enumerate(self.resblocks):
            if i == 0:
                x_ctrl = self.trainable_copies[0](x)
                x = block(x)
                x = x + self.zero_convs[0](x_ctrl)
            elif i < self.ctrl_layers:
                x_ctrl = self.trainable_copies[i](x_ctrl)
                x = block(x)
                x = x + self.zero_convs[i](x_ctrl)
            else:
                x = block(x)
        return x

class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
        use_pos_emb: bool = False,
        pos_emb_init_scale: float = 1.0,
        pos_emb_n_ctx: Optional[int] = None,
    ):
        super().__init__()
        self.controlnet_cond = False
        self.simple_blend = False
        self.device = device
        self.dtype = dtype
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.width = width
        self.updated_ctx = n_ctx + int(time_token_cond)
        self.time_token_cond = time_token_cond
        self.use_pos_emb = use_pos_emb
        self.init_scale = init_scale
        self.heads = heads
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        #self.backbone = Transformer(
        #    device=device,
        #    dtype=dtype,
        #    n_ctx=n_ctx + int(time_token_cond),
        #    width=width,
        #    layers=layers,
        #    heads=heads,
        #    init_scale=init_scale,
        #)
        self.backbone = TransformerControl(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            n_ctrl_layers=0,
            init_scale=init_scale,
        )
        #self.backbone = TransformerReverseControl(
        #    device=device,
        #    dtype=dtype,
        #    n_ctx=n_ctx + int(time_token_cond),
        #    width=width,
        #    layers=layers,
        #    heads=heads,
        #    n_ctrl_layers=0,
        #    init_scale=init_scale,
        #)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()
        if self.use_pos_emb:
            self.register_parameter(
                "pos_emb",
                nn.Parameter(
                    pos_emb_init_scale
                    * torch.randn(pos_emb_n_ctx or self.n_ctx, width, device=device, dtype=dtype)
                ),
            )

    def set_up_controlnet_cond(self):
        self.controlnet_cond = True

    def make_condition_head(self, simple_blend=False):
        self.simple_blend = simple_blend
        if simple_blend:
            self.raw_weights = torch.nn.Parameter(torch.randn(self.input_channels, 
                                                              self.width,
                                                              device=self.device,
                                                              dtype=self.dtype))
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.zero_conv_cond = nn.Conv1d(in_channels=self.input_channels, 
                                    out_channels=self.input_channels, 
                                    kernel_size=1,
                                    device=self.device,
                                    dtype=self.dtype)
            with torch.no_grad():
                self.zero_conv_cond.weight.zero_()
                self.zero_conv_cond.bias.zero_()

        self.cond_proj = nn.Linear(self.updated_ctx, 
                                   self.input_channels, 
                                   device=self.device, 
                                   dtype=self.dtype)
        
        self.cond_encoder_attn = ResidualAttentionBlock(
                    device=self.device,
                    dtype=self.dtype,
                    n_ctx=self.updated_ctx,
                    width=self.width,
                    heads=self.heads,
                    init_scale=self.init_scale,
                )
        

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]], **kwargs
    ) -> torch.Tensor:
        # set up condition
        if 'cond' in kwargs:
            h2 = self.input_proj(kwargs['cond'].reshape(x.shape).permute(0, 2, 1))
            if self.simple_blend:
                extra_tokens = [
                    (emb[:, None] if len(emb.shape) == 2 else emb)
                    for emb, as_token in cond_as_token
                    if as_token
                ]
                if len(extra_tokens):
                    raw = self.raw_weights.unsqueeze(dim=0).repeat(h2.shape[0], 1, 1)
                    raw = torch.cat(extra_tokens + [raw], dim=1)
                raw = self.cond_encoder_attn(raw)
                raw = self.cond_proj(raw.permute(0, 2, 1))
                blend_weights = self.sigmoid(raw)
            else:
                for emb, as_token in cond_as_token:
                    if not as_token:
                        h2 = h2 + emb[:, None]
                if self.use_pos_emb:
                    h2 = h2 + self.pos_emb
                extra_tokens = [
                    (emb[:, None] if len(emb.shape) == 2 else emb)
                    for emb, as_token in cond_as_token
                    if as_token
                ]
                if len(extra_tokens):
                    h2 = torch.cat(extra_tokens + [h2], dim=1)
                if self.controlnet_cond:
                    h2 = self.ln_pre(h2)
                    kwargs['cond_prepared'] = h2
                else:
                    h2 = self.cond_encoder_attn(h2)
                    h2 = self.cond_proj(h2.permute(0, 2, 1))
        
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        if 'cond' in kwargs:
            if self.simple_blend:
                h = h * (1.0 - blend_weights) + h2 * blend_weights
                self.total_update = blend_weights.detach().abs().sum().item()
            elif not self.controlnet_cond:
                update = self.zero_conv_cond(h2)
                h = h + update
                self.total_update = update.detach().abs().sum().item()
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        if self.use_pos_emb:
            h = h + self.pos_emb
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h, **kwargs)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)


class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        token_cond: bool = False,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        **kwargs,
    ):
        super().__init__(
            device=device, dtype=dtype, n_ctx=n_ctx + int(token_cond), pos_emb_n_ctx=n_ctx, **kwargs
        )
        self.n_ctx = n_ctx
        self.token_cond = token_cond
        self.clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(device)
        self.clip_embed = nn.Linear(
            self.clip.feature_dim, self.backbone.width, device=device, dtype=dtype
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            #return dict(embeddings=self.clip(batch_size, **model_kwargs))
            return dict(embeddings=self.clip(batch_size, texts=model_kwargs['texts']))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
        **kwargs,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param texts: a batch of texts to condition on.
        :param embeddings: a batch of CLIP embeddings to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        clip_out = self.clip(batch_size=len(x), images=images, texts=texts, embeddings=embeddings)
        assert len(clip_out.shape) == 2 and clip_out.shape[0] == x.shape[0]

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None].to(clip_out)

        # Rescale the features to have unit variance
        clip_out = math.sqrt(clip_out.shape[1]) * clip_out

        clip_embed = self.clip_embed(clip_out)

        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]
        return self._forward_with_cond(x, cond, **kwargs)


class CLIPImageGridPointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(device)
        super().__init__(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + clip.grid_size**2,
            pos_emb_n_ctx=n_ctx,
            **kwargs,
        )
        self.n_ctx = n_ctx
        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        _ = batch_size
        with torch.no_grad():
            return dict(embeddings=self.clip.embed_images_grid(model_kwargs["images"]))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C' x T] tensor.
        """
        assert images is not None or embeddings is not None, "must specify images or embeddings"
        assert images is None or embeddings is None, "cannot specify both images and embeddings"
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        else:
            clip_out = embeddings

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond), (clip_embed, True)]
        return self._forward_with_cond(x, cond)


class UpsamplePointDiffusionTransformer(PointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cond_input_channels: Optional[int] = None,
        cond_ctx: int = 1024,
        n_ctx: int = 4096 - 1024,
        channel_scales: Optional[Sequence[float]] = None,
        channel_biases: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + cond_ctx, **kwargs)
        self.n_ctx = n_ctx
        self.cond_input_channels = cond_input_channels or self.input_channels
        self.cond_point_proj = nn.Linear(
            self.cond_input_channels, self.backbone.width, device=device, dtype=dtype
        )

        self.register_buffer(
            "channel_scales",
            torch.tensor(channel_scales, dtype=dtype, device=device)
            if channel_scales is not None
            else None,
        )
        self.register_buffer(
            "channel_biases",
            torch.tensor(channel_biases, dtype=dtype, device=device)
            if channel_biases is not None
            else None,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, low_res: torch.Tensor):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)
        cond = [(t_embed, self.time_token_cond), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)

    def _embed_low_res(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_scales is not None:
            x = x * self.channel_scales[None, :, None]
        if self.channel_biases is not None:
            x = x + self.channel_biases[None, :, None]
        return self.cond_point_proj(x.permute(0, 2, 1))


class CLIPImageGridUpsamplePointDiffusionTransformer(UpsamplePointDiffusionTransformer):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int = 4096 - 1024,
        cond_drop_prob: float = 0.0,
        frozen_clip: bool = True,
        **kwargs,
    ):
        clip = (FrozenImageCLIP if frozen_clip else ImageCLIP)(device)
        super().__init__(device=device, dtype=dtype, n_ctx=n_ctx + clip.grid_size**2, **kwargs)
        self.n_ctx = n_ctx

        self.clip = clip
        self.clip_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.clip.grid_feature_dim,), device=device, dtype=dtype
            ),
            nn.Linear(self.clip.grid_feature_dim, self.backbone.width, device=device, dtype=dtype),
        )
        self.cond_drop_prob = cond_drop_prob

    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        _ = batch_size
        with torch.no_grad():
            return dict(
                embeddings=self.clip.embed_images_grid(model_kwargs["images"]),
                low_res=model_kwargs["low_res"],
            )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        low_res: torch.Tensor,
        images: Optional[Iterable[ImageType]] = None,
        embeddings: Optional[Iterable[torch.Tensor]] = None,
    ):
        """
        :param x: an [N x C1 x T] tensor.
        :param t: an [N] tensor.
        :param low_res: an [N x C2 x T'] tensor of conditioning points.
        :param images: a batch of images to condition on.
        :param embeddings: a batch of CLIP latent grids to condition on.
        :return: an [N x C3 x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        low_res_embed = self._embed_low_res(low_res)

        if images is not None:
            clip_out = self.clip.embed_images_grid(images)
        else:
            clip_out = embeddings

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            clip_out = clip_out * mask[:, None, None].to(clip_out)

        clip_out = clip_out.permute(0, 2, 1)  # NCL -> NLC
        clip_embed = self.clip_embed(clip_out)

        cond = [(t_embed, self.time_token_cond), (clip_embed, True), (low_res_embed, True)]
        return self._forward_with_cond(x, cond)
