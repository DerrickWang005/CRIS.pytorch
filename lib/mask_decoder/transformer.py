from typing import List

import fvcore.nn.weight_init as weight_init
import torch
import torchshow as ts
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry

from ..utils import (
    MLP,
    CrossAttentionLayer,
    FFNLayer,
    PositionEmbeddingRandom,
    PositionEmbeddingSine,
    SelfAttentionLayer,
)

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.CRIS.TRANSFORMER_DECODER_NAME
    in_channels = cfg.MODEL.CRIS.CONVS_DIM
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels)


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        *,
        embed_dim: int,
        nheads: int,
        activation: str,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        cls_dim: int,
        enforce_input_project: bool,
        num_query: int,
        mask_layers: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        # positional encoding
        N_steps = embed_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.embed_dim = embed_dim
        self.mask_dim = mask_dim
        self.cls_dim = cls_dim
        self.dim_feedforward = dim_feedforward
        self.num_heads = nheads
        self.pre_norm = pre_norm
        self.num_layers = dec_layers
        self.num_query = num_query

        # object query
        self.word_proj = nn.Linear(cls_dim, embed_dim)
        self.obj_query = nn.Embedding(num_query, embed_dim)
        obj_mask = torch.zeros(1, num_query).bool()
        self.register_buffer("obj_mask", obj_mask)

        # void
        self.void_embed = nn.Embedding(1, cls_dim)

        # transformer layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=embed_dim,
                    nhead=nheads,
                    dropout=0.0,
                    activation=activation,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=embed_dim,
                    nhead=nheads,
                    dropout=0.0,
                    activation=activation,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=embed_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    activation=activation,
                    normalize_before=pre_norm,
                )
            )

        # extra self-attention layer for learnable query features
        self.extra_self_attention_layer = SelfAttentionLayer(
            d_model=embed_dim,
            nhead=nheads,
            dropout=0.0,
            activation=activation,
            normalize_before=pre_norm,
        )
        # extra conv layer for mask features
        self.feat_embed = nn.Conv2d(embed_dim, mask_dim, 1, bias=True)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, embed_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != embed_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, embed_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # mask branch
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.mask_embed = MLP(embed_dim, embed_dim, mask_dim, mask_layers)
        self.cls_embed = MLP(embed_dim, embed_dim, cls_dim, mask_layers)

    @classmethod
    def from_config(cls, cfg, in_channels):
        ret = {}
        ret["in_channels"] = in_channels
        ret["embed_dim"] = cfg.MODEL.CRIS.EMBED_DIM
        ret["nheads"] = cfg.MODEL.CRIS.NHEADS
        ret["activation"] = cfg.MODEL.CRIS.ACTIVATION
        ret["dim_feedforward"] = cfg.MODEL.CRIS.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.CRIS.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.CRIS.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.CRIS.PRE_NORM
        ret["mask_dim"] = cfg.MODEL.CRIS.MASK_DIM
        ret["cls_dim"] = cfg.MODEL.CRIS.CLS_DIM
        ret["enforce_input_project"] = cfg.MODEL.CRIS.ENFORCE_INPUT_PROJ
        ret["num_query"] = cfg.MODEL.CRIS.NUM_QUERY
        ret["mask_layers"] = cfg.MODEL.CRIS.MASK_LAYERS

        return ret

    def forward(
        self,
        x: List[torch.Tensor],
        y_word: torch.Tensor,
        y_sent: torch.Tensor,
        y_pad_mask: torch.Tensor,
        mask_features: torch.Tensor,
    ):
        B, C, H, W = mask_features.shape
        _, T = y_pad_mask.shape

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src, pos, size_list = [], [], []

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # flatten NxCxHxW to NxHWxC
            pos.append(self.pe_layer(x[i], None).flatten(2).permute(0, 2, 1))
            src.append(self.input_proj[i](x[i]).flatten(2).permute(0, 2, 1) + self.level_embed.weight[i][None, None, :])

        # prepare object query
        obj_query = self.obj_query.weight  # Q, C
        obj_query = obj_query.unsqueeze(0).repeat(B, 1, 1)  # B, Q, C
        obj_mask = self.obj_mask.repeat(B, 1)  # B, Q

        # prepare query
        y_word = self.word_proj(y_word)  # B, T, C
        output = torch.cat([obj_query, y_word], dim=1)  # B, Q + T, C
        query_pos_embed = output
        query_pad_mask = torch.cat([obj_mask, y_pad_mask], dim=1)  # B, Q + T
        query_pad_mask = query_pad_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, self.num_query + T, 1)  # B, Q + T, Q + T

        # prepare cls
        void_embed = self.void_embed.weight.repeat(B, 1)
        y_sent = torch.stack([y_sent, void_embed], dim=1)

        # prediction heads on learnable query features
        predictions_mask = []
        predictions_cls = []

        output = self.extra_self_attention_layer(
            output,
            query_pos=None,
            tgt_mask=query_pad_mask,
        )
        outputs_mask, outputs_cls, attn_mask = self.forward_prediction_heads(
            output,
            y_sent,
            mask_features,
            attn_mask_target_size=size_list[0],
        )
        predictions_mask.append(outputs_mask)
        predictions_cls.append(outputs_cls)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_pos_embed,
            )
            # attention: self-attention
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=query_pad_mask,
                tgt_key_padding_mask=None,
                query_pos=query_pos_embed,
            )
            # ffn
            output = self.transformer_ffn_layers[i](output)
            # predict
            (outputs_mask, outputs_cls, attn_mask) = self.forward_prediction_heads(
                output,
                y_sent,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            )
            predictions_mask.append(outputs_mask)
            predictions_cls.append(outputs_cls)

        assert len(predictions_mask) == self.num_layers + 1

        out = {
            "pred_masks": predictions_mask[-1],
            "pred_logits": predictions_cls[-1],
            "aux_outputs": self._set_aux_loss(predictions_mask, predictions_cls),
        }
        return out

    def forward_prediction_heads(
        self,
        output: torch.Tensor,
        y_sent: torch.Tensor,
        mask_features: torch.Tensor,
        attn_mask_target_size: List,
    ):
        T = output.shape[1] - self.num_query
        # drop y_word
        output = output[:, : self.num_query]
        output = self.decoder_norm(output)  # B, Q, C

        mask_embed = self.mask_embed(output)
        cls_embed = self.cls_embed(output)  # B, Q, D
        mask_features = self.feat_embed(mask_features)

        # mask branch
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        outputs_cls = torch.einsum("bqc,bkc->bqk", cls_embed, y_sent)

        # NOTE: prediction is of higher-resolution
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        # [B, Q, K, H, W] -> [B, Q, 1, H, W] -> [B, 1, Q, 1, H, W] -> [B, 1, Q, 1, H*W] -> [B, 1, Q, K, H*W] -> [B, 1, Q*K, H*W]
        attn_mask = attn_mask.sigmoid().ge(0.5)
        attn_mask_y = attn_mask.sum(dim=1, keepdim=True).ge(0.5).repeat(1, T, 1, 1)
        attn_mask = torch.cat([attn_mask, attn_mask_y], dim=1).flatten(-2)  # B, Q + T, HW
        attn_mask = ~attn_mask.unsqueeze(1).detach()  # B, 1, Q + T, HW

        return outputs_mask, outputs_cls, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, predictions_mask, predictions_cls):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_masks": mask, "pred_logits": cls} for mask, cls in zip(predictions_mask[:-1], predictions_cls[:-1])
        ]
