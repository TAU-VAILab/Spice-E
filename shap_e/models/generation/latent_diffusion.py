from typing import Any, Dict

import torch
import torch.nn as nn


class SplitVectorDiffusion(nn.Module):
    def __init__(self, *, device: torch.device, wrapped: nn.Module, n_ctx: int, d_latent: int):
        super().__init__()
        self.device = device
        self.n_ctx = n_ctx
        self.d_latent = d_latent
        self.wrapped = wrapped

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
    
    def prepare_ctrlnet_for_training(self, 
                                     use_cond: bool=True, 
                                     out_proj: bool=False, 
                                     full_backbone: bool=False,
                                     full_backbone_hard: bool=False):
        for name, param in self.named_parameters():
            if name.startswith('wrapped.backbone.ctrl_resblocks'):
                split_name = name.split('.')
                if split_name[4] == 'frozen_copy':
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif name.startswith('wrapped.backbone.ca_ctrl_blocks'):
                if full_backbone_hard:
                    param.requires_grad = True
                    continue
                split_name = name.split('.')
                if split_name[5] == 'c_q2':
                    param.requires_grad = True
                elif split_name[5] == 'zero_conv':
                    param.requires_grad = True
                elif split_name[5] == 'one_conv':
                    param.requires_grad = True
                elif split_name[4] == 'ln1_cond':
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif name.startswith('wrapped.backbone.zero_convs'):
                param.requires_grad = True
            elif name.startswith('wrapped.backbone.trainable_copies'):
                param.requires_grad = True
            elif name.startswith('wrapped.backbone.cond_encoder') and use_cond:
                param.requires_grad = True
            elif name.startswith('wrapped.backbone.zero_conv_cond') and use_cond:
                param.requires_grad = True
            elif name.startswith('wrapped.output_proj') and out_proj:
                param.requires_grad = True
            elif name.startswith('wrapped.ln_post') and out_proj:
                param.requires_grad = True
            elif name.startswith('wrapped.zero_conv_cond'):
                param.requires_grad = True
            elif name.startswith('wrapped.backbone') and full_backbone_hard:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        if full_backbone:
            self.wrapped.backbone.set_controlnet_full_backbone()

    
    def prepare_for_training_shap_e_texturing(self, num_layers=None, reverse=False):
        for name, param in self.wrapped.named_parameters():
            split_name = name.split('.')
            if split_name[0] == 'cond_encoder_attn':
                param.requires_grad = True
            elif split_name[0] == 'zero_conv_cond':
                param.requires_grad = True
            elif split_name[0] == 'cond_proj':
                param.requires_grad = True
            elif split_name[0] == 'ln_pre':
                param.requires_grad = True
            elif split_name[0] == 'raw_weights':
                param.requires_grad = True
            self.freeze_transformer_backbone(num_layers=num_layers, reverse=reverse)
                
    def print_parameter_status(self):
        for name, param in self.wrapped.named_parameters():
            print(f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}")

    def freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def freeze_time_embedding(self):
        for param in self.wrapped.time_embed.parameters():
            param.requires_grad = False
    
    def freeze_positional_embedding(self):
        self.wrapped.pos_emb.requires_grad = False
    
    def freeze_input_projection(self):
        for param in self.wrapped.input_proj.parameters():
            param.requires_grad = False

    def unfreeze_transformer_backbone(self, num_layers=None, reverse=False):
        total_num_layers = self.wrapped.backbone.layers
        if num_layers is None:
            num_layers = total_num_layers
        for name, param in self.wrapped.backbone.named_parameters():
            split_name = name.split('.')
            layer_num = int(split_name[1])
            if reverse:
                layer_num = total_num_layers - layer_num
            if layer_num >= num_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def unfreeze_output_projection(self):
        for param in self.wrapped.output_proj.parameters():
            param.requires_grad = True
        for param in self.wrapped.ln_post.parameters():
            param.requires_grad = True
    
