# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models import register_model


from collections import OrderedDict
import torch
from torch import set_autocast_xla_enabled

'''
Adapter Module
'''

class Adapter(nn.Module):
    def __init__(self, config=None, d_model=None, bottleneck=None, dropout=0.0, init_option="bert",
                 adapter_scalar="1.0", adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        # layer_norm
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        self.init_option = init_option

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)


        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)


        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def reinitialize(self):
        """Reinitialize Adapter parameters (only 'lora' init option is supported)"""
        with torch.no_grad():
            if self.init_option == "lora":
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                if self.down_proj.bias is not None:
                    nn.init.zeros_(self.down_proj.bias)
                if self.up_proj.bias is not None:
                    nn.init.zeros_(self.up_proj.bias)
            else:
                raise NotImplementedError("Reinitialization is only implemented for lora adapter")


    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        B, N, C = x.shape
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output



'''
Attention Module
'''
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., config=None):
        super().__init__()

        self.config = config # load in configuration
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # set mlp for K
        self.k_mlp1 = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                             init_option=config.ffn_adapter_init_option,
                             adapter_scalar=config.ffn_adapter_scalar,
                             adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                             )
        self.k_mlp2 = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                             init_option=config.ffn_adapter_init_option,
                             adapter_scalar=config.ffn_adapter_scalar,
                             adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                             )

        # set mlp for v
        self.v_mlp1 = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                             init_option=config.ffn_adapter_init_option,
                             adapter_scalar=config.ffn_adapter_scalar,
                             adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                             )
        self.v_mlp2 = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num,
                             init_option=config.ffn_adapter_init_option,
                             adapter_scalar=config.ffn_adapter_scalar,
                             adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                             )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def reinitialize_adapter(self, flag):
        """
        Reinitialize Adapter module parameters based on flag value:
          - flag==1: Reinitialize k_mlp1 and v_mlp1 parameters
          - flag==2: Reinitialize k_mlp2 and v_mlp2 parameters
        """
        if flag == 1:
            self.k_mlp1.reinitialize()
            self.v_mlp1.reinitialize()
        elif flag == 2:
            self.k_mlp2.reinitialize()
            self.v_mlp2.reinitialize()
        else:
            raise ValueError("Flag must be either 1 or 2.")

    def copy_adapter(self, flag):
        """
        Copy Adapter module parameters:
          - flag==1: Copy k_mlp2 and v_mlp2 parameters to k_mlp1 and v_mlp1
          - flag==2: Copy k_mlp1 and v_mlp1 parameters to k_mlp2 and v_mlp2
        """
        if flag == 1:
            self.k_mlp1.load_state_dict(self.k_mlp2.state_dict())
            self.v_mlp1.load_state_dict(self.v_mlp2.state_dict())
        elif flag == 2:
            self.k_mlp2.load_state_dict(self.k_mlp1.state_dict())
            self.v_mlp2.load_state_dict(self.v_mlp1.state_dict())
        else:
            raise ValueError("Flag must be either 1 or 2.")

    def forward(self, x, flag):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.config.adapter_placement in ['attention', 'both']:
            if flag == 1:
                if self.config.ffn_adapt:
                    if self.config.ffn_option == 'parallel':
                        adapt_k = self.k_mlp1(x, add_residual=False)
                        adapt_v = self.v_mlp1(x, add_residual=False)
                        k = k + adapt_k
                        v = v + adapt_v
                    elif self.config.ffn_option == 'sequential':
                        k = self.k_mlp1(k)
                        v = self.v_mlp1(v)
                    else:
                        raise ValueError("Invalid ffn_option for ffn_adapt")
            elif flag == 2:
                if self.config.ffn_adapt:
                    if self.config.ffn_option == 'parallel':
                        adapt_k = self.k_mlp2(x, add_residual=False)
                        adapt_v = self.v_mlp2(x, add_residual=False)
                        k = k + adapt_k
                        v = v + adapt_v
                    elif self.config.ffn_option == 'sequential':
                        k = self.k_mlp2(k)
                        v = self.v_mlp2(v)
                    else:
                        raise ValueError("Invalid ffn_option for ffn_adapt")
            else:
                raise ValueError("The incoming flag value is incorrect")

        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x









# class Block(nn.Module):
#
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
#         super().__init__()
#         self.config = config
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, config=self.config)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.layer_id = layer_id
#         self.fc1 = nn.Linear(dim, mlp_hidden_dim)
#         self.fc2 = nn.Linear(mlp_hidden_dim, dim)
#         self.act = act_layer()
#         self.mlp_drop = nn.Dropout(drop)
#
#     def reinitialize_adapter(self, flag):
#         self.attn.reinitialize_adapter(flag)
#
#
#     def copy_adapter(self, flag):
#         self.attn.copy_adapter(flag)
#
#
#     def forward(self, x, flag):
#         x = x + self.drop_path(self.attn(self.norm1(x), flag))
#
#
#         residual = x
#         x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
#         x = self.drop_path(self.mlp_drop(self.fc2(x)))
#
#         """
#          Change the inserted layers
#         """
#
#         x = residual + x
#         return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              config=self.config)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.layer_id = layer_id
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        self.mlp_adapter1 = None
        self.mlp_adapter2 = None
        self.block_end_adapter1 = None
        self.block_end_adapter2 = None

        adapter_common_args = {
            'config': self.config,
            'd_model': dim,
            'bottleneck': config.ffn_num,
            'dropout': 0.1,
            'init_option': config.ffn_adapter_init_option,
            'adapter_scalar': config.ffn_adapter_scalar,
            'adapter_layernorm_option': config.ffn_adapter_layernorm_option,
        }

        if self.config.adapter_placement in ['mlp', 'both']:
            self.mlp_adapter1 = Adapter(**adapter_common_args)
            self.mlp_adapter2 = Adapter(**adapter_common_args)

        if self.config.adapter_placement == 'block_end':
            self.block_end_adapter1 = Adapter(**adapter_common_args)
            self.block_end_adapter2 = Adapter(**adapter_common_args)

    def reinitialize_adapter(self, flag):
        self.attn.reinitialize_adapter(flag)
        if self.config.adapter_placement in ['mlp', 'both']:
            if flag == 1:
                self.mlp_adapter1.reinitialize()
            elif flag == 2:
                self.mlp_adapter2.reinitialize()
        if self.config.adapter_placement == 'block_end':
            if flag == 1:
                self.block_end_adapter1.reinitialize()
            elif flag == 2:
                self.block_end_adapter2.reinitialize()

    def copy_adapter(self, flag):
        self.attn.copy_adapter(flag)
        if self.config.adapter_placement in ['mlp', 'both']:
            if flag == 1:
                self.mlp_adapter1.load_state_dict(self.mlp_adapter2.state_dict())
            elif flag == 2:
                self.mlp_adapter2.load_state_dict(self.mlp_adapter1.state_dict())
        if self.config.adapter_placement == 'block_end':
            if flag == 1:
                self.block_end_adapter1.load_state_dict(self.block_end_adapter2.state_dict())
            elif flag == 2:
                self.block_end_adapter2.load_state_dict(self.block_end_adapter1.state_dict())

    def forward(self, x, flag):
        original_placement = self.config.adapter_placement

        adapter_interval = getattr(self.config, 'adapter_interval', 1)

        if self.layer_id % adapter_interval != 0:
            self.config.adapter_placement = 'none'

        x = x + self.drop_path(self.attn(self.norm1(x), flag))

        mlp_residual = x
        mlp_x = self.norm2(x)
        mlp_x = self.mlp_drop(self.act(self.fc1(mlp_x)))
        mlp_x = self.fc2(mlp_x)

        if self.config.adapter_placement in ['mlp', 'both']:
            if flag == 1:
                adapt_out = self.mlp_adapter1(mlp_x, add_residual=False)
            else:
                adapt_out = self.mlp_adapter2(mlp_x, add_residual=False)
            mlp_x = mlp_x + adapt_out

        x = mlp_residual + self.drop_path(self.mlp_drop(mlp_x))

        if self.config.adapter_placement == 'block_end':
            if flag == 1:
                x = self.block_end_adapter1(x)
            else:
                x = self.block_end_adapter2(x)

        self.config.adapter_placement = original_placement

        return x


class VisionTransformer(nn.Module):

    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        print("I'm using ViT with adapters.")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm#range(depth)

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            self.embeddings = nn.ParameterList(
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in range(depth)
                 ])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        # set task num
        self.flag = 0

    def reinitialize_adapter(self, flag):
        for idx, blk in enumerate(self.blocks):
            blk.reinitialize_adapter(flag)


    def copy_adapter(self, flag):
        for idx, blk in enumerate(self.blocks):
            blk.copy_adapter(flag)

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, flag):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x, flag)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            features = x[:,0]
            fmaps = x[:,1:]

            outcome = {
                "features":features,
                "fmaps":fmaps,
            }

        return outcome

    def forward(self, x):
        if self.flag%2 == 1:
            with torch.no_grad():
                tx = self.forward_features(x,flag=1)
            sx = self.forward_features(x,flag=2)

        else:
            with torch.no_grad():
                tx = self.forward_features(x,flag=2)
            sx = self.forward_features(x,flag=1)



        out = {
                "tfeatures":tx["features"].detach(),
                "sfeatures":sx["features"],
                "tfmaps":tx["fmaps"].detach(),
                "sfmaps":sx["fmaps"],
            }


        out["tfeatures"] = self.head(out["tfeatures"])
        out["sfeatures"] = self.head(out["sfeatures"])

        return out


def vit_base_patch16_224_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)

    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768 * 2]
            v_weight = qkv_weight[768 * 2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768 * 2]
            v_bias = qkv_bias[768 * 2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)

    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model

