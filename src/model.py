import copy
import math

import torch
import torch.nn.functional as F
from transformers.models.mamba.configuration_mamba import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer


class ClipReverie(torch.nn.Module):
    def __init__(
        self,
        patch_x=3,
        patch_y=3,
        default_w=0.6,
    ):
        super(ClipReverie, self).__init__()
        self.patch_x = patch_x
        self.patch_y = patch_y

        self.gelu = torch.nn.GELU()
        self.w = torch.nn.Parameter(torch.tensor(default_w))

        logits_dim = 1024

        image_embed_dim = 1024 * 4 + 768 * 1
        self.image_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(image_embed_dim, 768 * 6),
            torch.nn.LayerNorm(768 * 6),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768 * 6, 768 * 5),
            torch.nn.LayerNorm(768 * 5),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768 * 5, image_embed_dim),
        )
        self.image_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(image_embed_dim, 1024 * 4),
            torch.nn.LayerNorm(1024 * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024 * 4, logits_dim),
        )

        text_embed_dim = 1024 * 2 + 768 * 1
        self.text_mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(text_embed_dim, logits_dim),
        )

        patch_embed_dim = 1024
        patch_embed_dim_dinov2 = 1024
        self.encoder_patch_dinov2 = self.create_encoder(
            8, patch_embed_dim_dinov2, patch_embed_dim_dinov2 * 2, 0.1, 4
        )
        self.attn_pool_dinov2 = AttentionPooling(patch_embed_dim_dinov2)
        self.fc3100 = torch.nn.Linear(patch_embed_dim_dinov2, patch_embed_dim)
        patch_embed_dim_clip = 768
        self.encoder_patch_clip = self.create_encoder(
            8, patch_embed_dim_clip, patch_embed_dim_clip * 2, 0.1, 4
        )
        self.attn_pool_clip = AttentionPooling(patch_embed_dim_clip)
        self.fc3200 = torch.nn.Linear(patch_embed_dim_clip, patch_embed_dim)

        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.fc6000 = torch.nn.Linear(1024, 1024)

    def image_encoder(
        self,
        image_clip,
        gpt4o_stella,
        patch_clip,
        patch_dinov2,
        multilayer_dinov2,
    ):
        image_clip = image_clip.float()
        gpt4o_stella = gpt4o_stella.float()
        patch_clip = patch_clip.float()
        patch_dinov2 = patch_dinov2.float()
        multilayer_dinov2 = multilayer_dinov2.float()

        multilayer_st_embeddings = self.multilayer_st_encoder(
            multilayer_dinov2, gpt4o_stella
        )
        patch_clip = self.patch_encoder_clip(patch_clip)
        patch_dinov2 = self.patch_encoder_dinov2(patch_dinov2)

        image_embeddings = torch.cat(
            [
                gpt4o_stella,
                patch_dinov2,
                multilayer_st_embeddings,
                image_clip,
                patch_clip,
            ],
            dim=1,
        )
        identity = image_embeddings
        image_embeddings = self.image_mlp1(image_embeddings)
        image_embeddings = image_embeddings + identity
        image_embeddings = self.image_mlp2(image_embeddings)
        return image_embeddings

    def text_encoder(
        self,
        instruction_clip,
        instruction_stella,
        target_object_explanation_stella,
    ):
        instruction_clip = instruction_clip.float()
        instruction_stella = instruction_stella.float()
        target_object_explanation_stella = target_object_explanation_stella.float()

        text_embeddings = torch.cat(
            [
                instruction_clip,
                instruction_stella,
                target_object_explanation_stella,
            ],
            dim=1,
        )
        text_embeddings = self.text_mlp(text_embeddings)
        return text_embeddings

    def patch_encoder_dinov2(self, patch_embeddings):
        patch_embeddings = self.encoder_patch_dinov2(patch_embeddings)
        patch_embeddings = self.attn_pool_dinov2(patch_embeddings)
        patch_embeddings = self.fc3100(patch_embeddings)
        return patch_embeddings

    def patch_encoder_clip(self, patch_embeddings):
        patch_embeddings = self.encoder_patch_clip(patch_embeddings)
        patch_embeddings = self.attn_pool_clip(patch_embeddings)
        patch_embeddings = self.fc3200(patch_embeddings)
        return patch_embeddings

    def scene_text_encoder(self, text_feats, ocr_info):
        pos_info = ocr_info[:, :, 1:-1]
        pos_embeddings = self.posencoder(pos_info)
        conf_input = torch.cat([text_feats, pos_info.float()], dim=-1)
        conf_scores = torch.sigmoid(self.ocr_confidence(conf_input))

        filtered_text_feats = text_feats * conf_scores
        combined_feats = filtered_text_feats + pos_embeddings
        transformed_feats = self.fc4000(combined_feats)

        attn_output, _ = self.st_attention(
            transformed_feats, transformed_feats, transformed_feats
        )
        encoder_output = self.encoder_scene_text(attn_output)
        aggregated = encoder_output[:, 0, :]
        scene_text_features = self.fc4001(aggregated)

        return scene_text_features

    def multilayer_st_encoder(self, multilayer_image_feats, image_desc_feats):
        query = image_desc_feats.unsqueeze(0)
        key = multilayer_image_feats.transpose(0, 1)
        value = key

        attn_output, _ = self.cross_attn(query, key, value)
        attn_output = attn_output.squeeze(0)
        output = self.fc6000(attn_output)

        return output

    def calc_token_matching(self, ocr_tokens, ne_tokens, chunk_size=128):
        ocr_tokens_exp = ocr_tokens.unsqueeze(0).unsqueeze(-1)
        ocr_nonzero = ocr_tokens_exp != 0

        def compute_logits(ne_chunk):
            ne_chunk_exp = ne_chunk.unsqueeze(1).unsqueeze(-2)
            ne_nonzero = ne_chunk_exp != 0
            eq = (ocr_tokens_exp == ne_chunk_exp) & ocr_nonzero & ne_nonzero
            return eq.sum(dim=(-2, -1))

        if chunk_size >= ne_tokens.size(0):
            logits = compute_logits(ne_tokens)
            return logits
        else:
            logits_list = []
            for i in range(0, ne_tokens.size(0), chunk_size):
                ne_chunk = ne_tokens[i : i + chunk_size]
                logits_chunk = compute_logits(ne_chunk)
                logits_list.append(logits_chunk)
            logits = torch.cat(logits_list, dim=0)
            return logits

    def forward(
        self,
        instruction_clip,
        image_clip,
        instruction_stella,
        target_object_explanation_stella,
        gpt4o_stella,
        patch_clip,
        patch_dinov2,
        multilayer_dinov2,
        ne_tokens,
        ocr_tokens,
    ):
        token_logits = self.calc_token_matching(ocr_tokens, ne_tokens)
        token_logits = token_logits.float()
        token_logits = F.normalize(token_logits, p=2, dim=1)

        image_embeddings = self.image_encoder(
            image_clip,
            gpt4o_stella,
            patch_clip,
            patch_dinov2,
            multilayer_dinov2,
        )
        text_embeddings = self.text_encoder(
            instruction_clip,
            instruction_stella,
            target_object_explanation_stella,
        )
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        logits = text_embeddings @ image_embeddings.T

        logits = logits * self.w + token_logits * (1 - self.w)
        return logits

    def create_encoder(self, h, d_model, d_ff, dropout, N):
        mamba_config = MambaConfig(
            hidden_size=d_model,
            state_size=8,
            conv_kernel=4,
            expand=1.5,
            use_bias=True,
            use_conv_bias=True,
            hidden_act="silu",
        )
        config = MambaTransformerConfig(
            d_model, d_ff, dropout, N, mamba_config=mamba_config
        )
        return config.create_encoder()


class Encoder(torch.nn.Module):
    def __init__(self, N, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class SSMEncoder(torch.nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layers):
        super(SSMEncoder, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.norm = torch.nn.LayerNorm(layers[0].size)

    def forward(self, x):
        """Pass the input through each layer in turn."""
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


def clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(torch.nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class MambaEncoderLayer(torch.nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(MambaEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(torch.nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.parameter.Parameter(torch.ones(features))
        self.b_2 = torch.nn.parameter.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            l_fn(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l_fn, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class CrossAttentionEncoder(torch.nn.Module):
    def __init__(self, N, layer):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, query, kv):
        for layer in self.layers:
            query = layer(query, kv)
        return self.norm(query)


class CrossAttentionEncoderLayer(torch.nn.Module):
    def __init__(self, size, cross_attn, feed_forward, dropout):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, kv):
        x = self.sublayer[0](x, lambda x: self.cross_attn(x, kv, kv))
        return self.sublayer[1](x, self.feed_forward)


class CrossAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(CrossAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            l_fn(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l_fn, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class CrossTransformerConfig:
    def __init__(self, h, d_model, d_ff, dropout, N):
        self.h = h
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.N = N

    def create_encoder(self):
        c = copy.deepcopy
        cross_attn = CrossAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        return CrossAttentionEncoder(
            self.N,
            CrossAttentionEncoderLayer(
                self.d_model, c(cross_attn), c(ff), dropout=self.dropout
            ),
        )


class TransformerConfig:
    def __init__(self, h, d_model, d_ff, dropout, N):
        self.h = h
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.N = N

    def create_encoder(self):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        return Encoder(
            self.N, EncoderLayer(self.d_model, c(attn), c(ff), dropout=self.dropout)
        )


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttentionPooling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        attn_weights = torch.softmax(query @ x.transpose(1, 2), dim=-1)
        pooled = attn_weights @ x
        return self.proj(pooled.squeeze(1))


class MambaTransformerConfig:
    """Transformer settings with MambaMixer"""

    def __init__(self, d_model, d_ff, dropout, N, mamba_config):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.N = N
        self.mamba_config = mamba_config or MambaConfig(
            hidden_size=d_model,
            state_size=16,
            conv_kernel=4,
            expand=2,
            use_bias=True,
            use_conv_bias=True,
            hidden_act="silu",
        )

    def create_encoder(self):
        layers = []
        for i in range(self.N):
            drop = (i + 1) / self.N
            attn = MambaMixer(config=self.mamba_config, layer_idx=i)
            ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
            layers.append(MambaEncoderLayer(self.d_model, attn, ff, dropout=drop))
        return SSMEncoder(layers)
