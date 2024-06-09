import open_clip
import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils import comm


def build_backbone(cfg):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_name = cfg.MODEL.BACKBONE.CLIP_MODEL_NAME
    pretrained = cfg.MODEL.BACKBONE.CLIP_PRETRAINED_WEIGHTS
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, model_name, pretrained)
    assert isinstance(backbone, Backbone)
    return backbone


@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    @configurable
    def __init__(self, model_name, pretrained):
        super().__init__()
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.model_name = model_name
        self.pretrained = pretrained
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        model_name = model_name.lower()
        assert "convnext_" in model_name, "Only convnext models are supported"
        self.model_type = "convnext"
        if "_base" in model_name:
            self.output_channels = [128, 128, 256, 512, 1024]
        elif "_large" in model_name:
            self.output_channels = [192, 192, 384, 768, 1536]
        elif "_xxlarge" in model_name:
            self.output_channels = [384, 384, 768, 1536, 3072]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1,
        }
        self._out_feature_channels = {
            "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent,
        }
        # self.freeze_everything()
        # self.freeze_visual()
        # self.freeze_text()
        self.no_freeze()

    @classmethod
    def from_config(cls, cfg, model_name, pretrained):
        ret = {}
        ret["model_name"] = model_name
        ret["pretrained"] = pretrained

        return ret

    def freeze_everything(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def freeze_visual(self):
        for name, param in self.named_parameters():
            if "visual" in name:
                param.requires_grad = False
            if "logit_scale" in name:
                param.requires_grad = False

    def freeze_text(self):
        for name, param in self.named_parameters():
            if "visual" not in name:
                param.requires_grad = False
            if "logit_scale" in name or "visual.trunk" in name or "visual.head" in name:
                param.requires_grad = False

    def no_freeze(self):
        for name, param in self.named_parameters():
            if "logit_scale" in name or "visual.trunk" in name or "visual.head" in name:
                param.requires_grad = False

    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def get_text_projection(self, x):
        return x @ self.clip_model.text_projection

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eos = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(eos, dim=-1) if normalize else eos, x

    # @torch.no_grad()
    def extract_text_features(self, text_list, device):
        # self.eval()
        # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
        text_tokens = self.tokenize_text(text_list)
        text_tokens = text_tokens.to(device)
        pad_mask = text_tokens == 0
        # we return un-normalized text feature.
        sent_features, text_features = self.encode_text(text_tokens, normalize=False)
        return sent_features, text_features, pad_mask

    # @torch.no_grad()
    def extract_visual_features(self, x):
        # self.eval()
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out["stem"] = x.contiguous()  # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f"res{i+2}"] = x.contiguous()  # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        x = self.clip_model.visual.trunk.norm_pre(x)
        out["clip_vis_dense"] = x.contiguous()
        return out

    def visual_prediction_forward(self, x, masks=None):
        batch, num_query, channel = x.shape
        x = x.reshape(batch * num_query, channel, 1, 1)  # fake 2D input
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1])  # B x num_queries x 640

    def forward(self, x):
        raise NotImplementedError

    @property
    def dim_latent(self):
        return self.clip_model.text_projection.shape[-1]

    @property
    def size_divisibility(self):
        return -1

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
        }
