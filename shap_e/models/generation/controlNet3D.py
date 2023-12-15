from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from .pretrained_clip import FrozenImageCLIP, ImageCLIP, ImageType
from .transformer import MLP, timestep_embedding, ResidualAttentionBlock

import math
import torch
import torch.nn as nn

def copy_weights_from_shap_e(shap_e_model: nn.Module, ctrl3d_model: nn.Module):
    num_control_layers = ctrl3d_model.n_ctrol_layers
    num_reg_layers = ctrl3d_model.layers - num_control_layers
    for name, param in shap_e_model.named_parameters():
        if name in ctrl3d_model.state_dict().keys():
            ctrl3d_model.state_dict()[name].copy_(param)
        else:
            split_name = name.split('.')
            # this should be the case for all control resblocks params
            if split_name[2] == 'resblocks':
                ctrol_block_num = int(split_name[3]) - num_reg_layers
                frozen_name = f'wrapped.backbone.ctrl_resblocks.{ctrol_block_num}.frozen_copy.{".".join(split_name[4:])}'
                assert frozen_name in ctrl3d_model.state_dict().keys(), f"param {frozen_name} not found in ctrl3d_model"
                trainable_name = f'wrapped.backbone.ctrl_resblocks.{ctrol_block_num}.trainable_copy.{".".join(split_name[4:])}'
                assert trainable_name in ctrl3d_model.state_dict().keys(), f"param {trainable_name} not found in ctrl3d_model"
                ctrl3d_model.state_dict()[frozen_name].copy_(param)
                ctrl3d_model.state_dict()[trainable_name].copy_(param)


class ResidualAttentionBlockControl(nn.Module):
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

        self.frozen_copy = ResidualAttentionBlock(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale
        )
        self.trainable_copy = ResidualAttentionBlock(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale
        )
        self.zero_conv_out = nn.Conv1d(in_channels=n_ctx, 
                                    out_channels=n_ctx, 
                                    kernel_size=1,
                                    device=device,
                                    dtype=dtype)
        with torch.no_grad():
            self.zero_conv_out.weight.zero_()
            self.zero_conv_out.bias.zero_()
        
    def forward(self, x: torch.Tensor):
        x_frozen, x_trainable = torch.chunk(x, 2, dim=1)
        x_trainable = self.trainable_copy(x_trainable)
        x_frozen = self.frozen_copy(x_frozen)
        x_frozen = x_frozen + self.zero_conv_out(x_trainable)
        x = torch.cat([x_frozen, x_trainable], dim=1)
        return x

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
        n_ctrl_layers: int = 12,
        init_scale: float = 0.25,
    ):
        super().__init__()
        assert n_ctrl_layers <= layers, \
            f"n_ctrl_layers ({n_ctrl_layers}) must be <= layers ({layers})"
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
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
                for _ in range(self.reg_layers)
            ]
        )
        self.ctrl_resblocks = nn.ModuleList(
            [
                ResidualAttentionBlockControl(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(self.ctrl_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        
        # TODO (ES): Control condition should go here with zero conv
        x_dub = torch.cat([x, x], dim=1)
        for block in self.ctrl_resblocks:
            x_dub = block(x_dub)
        x_res, _ = torch.chunk(x_dub, 2, dim=1)
        return x_res

class SplitVectorDiffusionControl(nn.Module):
    def __init__(self, 
                 *, 
                 device: torch.device, 
                 dtype: torch.dtype, 
                 n_ctx: int, 
                 d_latent: int, 
                 n_ctrol_layers: int,
                 width: int = 512,
                 layers: int = 24,
                 heads: int = 8,
                 init_scale: float = 0.25,
                 time_token_cond: bool = True,
                 use_pos_emb: bool = True,
                 pos_emb_init_scale: float = 0.05,
                 pos_emb_n_ctx: Optional[int] = 1024,
            ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_ctx = n_ctx
        self.d_latent = d_latent
        self.layers = layers
        self.n_ctrol_layers = n_ctrol_layers
        self.wrapped = CLIPImagePointDiffusionTransformerControl(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            time_token_cond=time_token_cond,
            use_pos_emb=use_pos_emb,
            pos_emb_init_scale=pos_emb_init_scale,
            pos_emb_n_ctx=pos_emb_n_ctx,
            n_ctrl_layers=n_ctrol_layers,
        )

        if hasattr(self.wrapped, "cached_model_kwargs"):
            self.cached_model_kwargs = self.wrapped.cached_model_kwargs

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        h = x.reshape(x.shape[0], self.n_ctx, -1).permute(0, 2, 1)
        pre_channels = h.shape[1]
        h = self.wrapped(h, t, **kwargs)
        assert (
            h.shape[1] == pre_channels * 2
        ), "expected twice as many outputs for variance prediction"
        eps, var = torch.chunk(h, 2, dim=1)
        return torch.cat(
            [
                eps.permute(0, 2, 1).flatten(1),
                var.permute(0, 2, 1).flatten(1),
            ],
            dim=1,
        )
    
    def prepare_for_training(self):
        for name, param in self.named_parameters():
            if name.startswith('wrapped.backbone.ctrl_resblocks'):
                split_name = name.split('.')
                if split_name[4] == 'frozen_copy':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False

    def print_parameter_status(self):
        print(f"====== Time Embedding Network ======")
        for name, param in self.wrapped.time_embed.named_parameters():
            print(f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}")
        print(f"====== Positional Embedding ======")
        print(f"name: pos_embed, shape: {self.wrapped.pos_emb.shape}, req grad: {self.wrapped.pos_emb.requires_grad}")
        print(f"====== Input Projection ======")
        for name, param in self.wrapped.input_proj.named_parameters():
            print(f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}")
        print(f"====== Transformer Backbone ======")
        for name, param in self.wrapped.backbone.named_parameters():
            print(f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}")
        print(f"====== Output Projection ======")
        for name, param in self.wrapped.output_proj.named_parameters():
            print(f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}")
    
    def freeze_time_embedding(self):
        for param in self.wrapped.time_embed.parameters():
            param.requires_grad = False
    
    def freeze_positional_embedding(self):
        self.wrapped.pos_emb.requires_grad = False
    
    def freeze_input_projection(self):
        for param in self.wrapped.input_proj.parameters():
            param.requires_grad = False

    def freeze_transformer_backbone(self, num_layers=None, reverse=False):
        total_num_layers = self.wrapped.backbone.layers
        if num_layers is None:
            num_layers = total_num_layers
        for name, param in self.wrapped.backbone.named_parameters():
            split_name = name.split('.')
            layer_num = int(split_name[1])
            if reverse:
                layer_num = total_num_layers - layer_num
            if layer_num < num_layers:
                param.requires_grad = False
    
    def freeze_output_projection(self):
        for param in self.wrapped.output_proj.parameters():
            param.requires_grad = False

class PointDiffusionTransformerControl(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 1024,
        output_channels: int = 2048,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 24,
        heads: int = 16,
        init_scale: float = 0.25,
        time_token_cond: bool = True,
        use_pos_emb: bool = True,
        pos_emb_init_scale: float = 0.05,
        n_ctrl_layers: int = 12,
        pos_emb_n_ctx: Optional[int] = 1024,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.use_pos_emb = use_pos_emb
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = TransformerControl(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            n_ctrl_layers=n_ctrl_layers,
        )
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
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
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
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)
    
class CLIPImagePointDiffusionTransformerControl(PointDiffusionTransformerControl):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 1024,
        output_channels: int = 2048,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 24,
        heads: int = 16,
        init_scale: float = 0.25,
        time_token_cond: bool = True,
        use_pos_emb: bool = True,
        pos_emb_init_scale: float = 0.05,
        n_ctrl_layers: int = 12,
        pos_emb_n_ctx: Optional[int] = 1024,
        token_cond: bool = True,
        cond_drop_prob: float = 0.1,
        frozen_clip: bool = True,
        **kwargs,
    ):
        super().__init__(
            device=device, 
            dtype=dtype, 
            n_ctx=n_ctx + int(token_cond),
            input_channels=input_channels,
            output_channels=output_channels,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            time_token_cond=time_token_cond,
            use_pos_emb=use_pos_emb,
            pos_emb_init_scale=pos_emb_init_scale,
            pos_emb_n_ctx=pos_emb_n_ctx, 
            n_ctrl_layers=n_ctrl_layers, 
            **kwargs
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
            return dict(embeddings=self.clip(batch_size, **model_kwargs))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        images: Optional[Iterable[Optional[ImageType]]] = None,
        texts: Optional[Iterable[Optional[str]]] = None,
        embeddings: Optional[Iterable[Optional[torch.Tensor]]] = None,
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

        with torch.no_grad():
            clip_embed = self.clip_embed(clip_out)

        cond = [(clip_embed.detach(), self.token_cond), (t_embed.detach(), self.time_token_cond)]
        res = self._forward_with_cond(x, cond)
        del cond
        return res