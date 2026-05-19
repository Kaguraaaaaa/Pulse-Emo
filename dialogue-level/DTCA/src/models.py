import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig, AutoModel
from typing import Optional
from utils import to_gpu
from utils import ReverseLayerF
import warnings
from transformers import RobertaTokenizer, RobertaModel

import math
import torch.nn.functional as F
import torch.nn as disable_weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class AdversarialDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(AdversarialDiscriminator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

class RNNPoolingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNPoolingModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [frame, batch, embedding]
        _, (h_n, _) = self.rnn(x)  # h_n: [1, batch, hidden_dim]
        x = h_n.squeeze(0)  # [batch, hidden_dim]
        x = self.fc(x)  # [batch, output_dim]
        return x
    
class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)  

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        return attn_output
    
    
def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=8):
        super().__init__()
        self.dim_q = dim_q
        self.heads = heads
        self.wq = nn.Linear(dim_q, dim_q, bias=False)
        self.wk = nn.Linear(dim_kv, dim_q, bias=False)
        self.wv = nn.Linear(dim_kv, dim_q, bias=False)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU
    def multihead_reshape(self, x):
        b, lens, dim = x.shape
        x = x.reshape(b, lens, self.heads, dim // self.heads)
        x = x.transpose(1, 2)
        x = x.reshape(b * self.heads, lens, dim // self.heads)
        return x

    def multihead_reshape_inverse(self, x):
        b, lens, dim = x.shape
        x = x.reshape(b // self.heads, self.heads, lens, dim)
        x = x.transpose(1, 2)
        x = x.reshape(b // self.heads, lens, dim * self.heads)
        return x
    def forward(self, q, kv):
        q = self.wq(q)
        k = self.wk(kv)
        v = self.wv(kv)

        q = self.multihead_reshape(q)
        k = self.multihead_reshape(k)
        v = self.multihead_reshape(v)

        atten = q.bmm(k.transpose(1, 2)) * (self.dim_q // self.heads)**-0.5
        atten = atten.softmax(dim=-1)
        atten = atten.bmm(v)
        
        atten = self.multihead_reshape_inverse(atten)
        atten = self.out_proj(atten)
        atten = self.dropout(atten)
        return atten
    
class FBP(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, fbp_hid, fbp_k, dropout):
        super(FBP, self).__init__()
        self.fusion_1_matrix = nn.Linear(d_emb_1, fbp_hid*fbp_k, bias=False)
        self.fusion_2_matrix = nn.Linear(d_emb_2, fbp_hid*fbp_k, bias=False)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_pooling = nn.AvgPool1d(kernel_size=fbp_k)
        self.fbp_k = fbp_k

    def forward(self, seq1, seq2):
        seq1 = self.fusion_1_matrix(seq1)
        seq2 = self.fusion_2_matrix(seq2)
        fused_feature = torch.mul(seq1, seq2)
        if len(fused_feature.shape) == 2:
            fused_feature = fused_feature.unsqueeze(0)
        fused_feature = self.fusion_dropout(fused_feature)
        fused_feature = self.fusion_pooling(fused_feature).squeeze(0) * self.fbp_k # (bs, 512)
        fused_feature = F.normalize(fused_feature, dim=-1, p=2)
        return fused_feature
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
            nn.Dropout(0.6)
        )
    
    def forward(self, x):
        gate_value = self.gate(x)
        return x * gate_value
def compute_frame_attention(self, features_m1, features_m2):
    S_g_list = []
    for frame in range(features_m1.size(0)):
        Q = features_m1[frame]  #  [batch, ebd]
        K = features_m2[frame]  #  [batch, ebd]
        elementwise_mul = Q * K  #  [batch, ebd]

        # SumPooling
        sum_pooled = torch.sum(elementwise_mul, dim=1)  #  [batch]

        # L2 Normalization
        l2_normalized = F.normalize(sum_pooled, p=2, dim=1)  #  [batch]

        # Linear Projection
        S_g = torch.matmul(l2_normalized)  # [batch]
        
        S_g_list.append(S_g)
    S_g_final = torch.stack(S_g_list, dim=0)  #  [frame, batch]
    return S_g_final

class MyRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MyRNNModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        _, (hn, _) = self.rnn(x)
        hn = hn[-1] 
        return hn
    
# let's define a simple model that can deal with multimodal variable length sequence

class LoRALinear(nn.Module):
    """
    A minimal LoRA wrapper for nn.Linear.

    Replaces: y = xW^T (+ b)  ->  y = xW^T (+ b) + (dropout(x) A B) * (alpha/r)
    Where:
      - A: (in_features, r)
      - B: (r, out_features)
    """

    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze original weights
        self.base_weight = base_linear.weight
        self.base_weight.requires_grad_(False)
        if base_linear.bias is not None:
            self.base_bias = base_linear.bias
            self.base_bias.requires_grad_(False)
        else:
            self.base_bias = None

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(self.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is initialized to zeros by default (already zeros above)

    def forward(self, x):
        # x: [..., in_features]
        result = F.linear(x, self.base_weight, self.base_bias)
        lora_inter = self.lora_dropout(x) @ self.lora_A  # [..., r]
        lora_update = lora_inter @ self.lora_B  # [..., out_features]
        return result + lora_update * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_keywords,
):
    """
    Replace Linear modules whose qualified name contains any keyword.
    Example keywords: ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'].
    """
    if isinstance(target_keywords, str):
        target_keywords = [s.strip() for s in target_keywords.split(",") if s.strip()]
    target_keywords = list(target_keywords)

    # Collect first to avoid mutating while iterating
    to_replace = []
    for name, module in model.named_modules():
        if any(kw in name for kw in target_keywords) and isinstance(module, nn.Linear):
            to_replace.append((name, module))

    for name, module in to_replace:
        parts = name.split(".")
        attr = parts[-1]
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))

    return model


class SATI_QwenLoRA(nn.Module):
    """
    Qwen + (multi-modal prefix tuning) + classification head.

    Strategy:
      - Text: token embeddings from Qwen input_ids
      - Visual/Acoustic/Physio: pool variable-length sequences into K prefix tokens,
        project them into Qwen hidden size, then prepend as extra tokens.
      - Train only LoRA parameters + modality projectors + classifier.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size
        self.physio_size = getattr(config, 'physio_size', 0) 
        self.num_classes = int(getattr(config, "num_classes", 7))
        self.dropout_rate = float(getattr(config, "dropout", 0.1))
        self.activation = self.config.activation()
        # Qwen backbone
        self.llm_type = getattr(config, "llm_type", "qwen").lower()
        if self.llm_type != "qwen":
            raise ValueError("SATI_QwenLoRA requires llm_type='qwen'")

        self.llm_path = getattr(config, "llm_path", "/data/s22xjq/Qwen")
        self.llm = AutoModel.from_pretrained(self.llm_path, trust_remote_code=True)
        self.llm_hidden_size = getattr(self.llm.config, "hidden_size", None)
        if self.llm_hidden_size is None:
            raise ValueError("Cannot infer llm_hidden_size from llm.config.hidden_size")
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # Freeze base LLM
        for p in self.llm.parameters():
            p.requires_grad_(False)

        # Apply LoRA on attention/MLP Linear layers
        if getattr(config, "use_lora", True):
            target_modules = getattr(config, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
            self.llm = apply_lora_to_model(
                self.llm,
                r=int(getattr(config, "lora_r", 16)),
                alpha=int(getattr(config, "lora_alpha", 32)),
                dropout=float(getattr(config, "lora_dropout", 0.05)),
                target_keywords=target_modules,
            )

        # Modality sizes are injected by data_loader into config
        self.visual_size = int(getattr(config, "visual_size", 0))
        self.acoustic_size = int(getattr(config, "acoustic_size", 0))
        self.physio_size = int(getattr(config, "physio_size", 0))
        print("self.visual_size", self.visual_size)
        print("self.acoustic_size", self.acoustic_size)
        print("self.physio_size", self.physio_size)
        # Prefix token lengths per modality
        self.kv = int(getattr(config, "prefix_len_visual", 8))
        self.ka = int(getattr(config, "prefix_len_acoustic", 8))
        self.kp = int(getattr(config, "prefix_len_physio", 8))

        # Pooling modules: adaptive avg pool across time dimension
        self.pool_visual = nn.AdaptiveAvgPool1d(self.kv)
        self.pool_acoustic = nn.AdaptiveAvgPool1d(self.ka)
        self.pool_physio = nn.AdaptiveAvgPool1d(self.kp) if self.physio_size > 0 else None

        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]

        # Project pooled modality tokens into Qwen hidden space
        # self.visual_proj = nn.Linear(self.visual_size, self.llm_hidden_size) if self.visual_size > 0 else None
        # self.acoustic_proj = nn.Linear(self.acoustic_size, self.llm_hidden_size) if self.acoustic_size > 0 else None
        # self.physio_proj = nn.Linear(self.physio_size, self.llm_hidden_size) if self.physio_size > 0 else None
        self.visual_proj = nn.Linear(self.visual_size, self.llm_hidden_size) if self.visual_size > 0 else None
        self.acoustic_proj = nn.Linear(self.acoustic_size, self.llm_hidden_size) if self.acoustic_size > 0 else None
        self.physio_proj = nn.Linear(self.physio_size, self.llm_hidden_size) if self.physio_size > 0 else None

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.llm_hidden_size, self.num_classes),
        )

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1], out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2], out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


        self.project_p = nn.Sequential()
        self.project_p.add_module('project_p', nn.Linear(in_features=2*hidden_sizes[2], out_features=config.hidden_size))
        self.project_p.add_module('project_p_activation', self.activation)
        self.project_p.add_module('project_p_layer_norm', nn.LayerNorm(config.hidden_size))
        
        
        # self.prnn1 = rnn(self.physio_size, hidden_sizes[2], bidirectional=True)
        # self.prnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)
        
        
        
        # self.vrnn1 = rnn(self.visual_size, hidden_sizes[1], bidirectional=True)
        # self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        # self.arnn1 = rnn(self.acoustic_size, hidden_sizes[2], bidirectional=True)
        # self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        self.hidden_sizes = hidden_sizes = [
            int(self.text_size),
            int(self.visual_size),
            int(self.acoustic_size)
        ]

        # RNN encoders
        self.vrnn1 = rnn(self.visual_size, hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = rnn(self.acoustic_size, hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.prnn1 = rnn(self.physio_size, hidden_sizes[2], bidirectional=True)
        self.prnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        # Pooling
        self.pool_visual = nn.AdaptiveAvgPool1d(self.kv)
        self.pool_acoustic = nn.AdaptiveAvgPool1d(self.ka)
        self.pool_physio = nn.AdaptiveAvgPool1d(self.kp) if self.physio_size > 0 else None

        # Project RNN outputs to Qwen hidden size
        self.visual_proj = nn.Linear(2 * hidden_sizes[1], self.llm_hidden_size) if self.visual_size > 0 else None
        self.acoustic_proj = nn.Linear(2 * hidden_sizes[2], self.llm_hidden_size) if self.acoustic_size > 0 else None
        self.physio_proj = nn.Linear(2 * hidden_sizes[2], self.llm_hidden_size) if self.physio_size > 0 else None
    def _pool_seq(self, seq_tb_d, pool_k: int, pool_module: nn.Module):
        """
        seq_tb_d: [T, B, d_in]
        returns:  [B, k, d_in]
        """
        # -> [B, d_in, T]
        x = seq_tb_d.permute(1, 2, 0)
        # -> [B, d_in, k]
        x = pool_module(x)
        # -> [B, k, d_in]
        return x.permute(0, 2, 1)

    def _masked_mean(self, hidden, attn_mask):
        """
        hidden: [B, L, D], attn_mask: [B, L]
        """
        mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask, physio=None):
        input_ids = bert_sent
        attention_mask = bert_sent_mask

        token_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, T, D]

        prefix_tokens = []
        B = token_embeds.size(0)
        device = token_embeds.device
        
        
        video_rnn1_out, _ = self.vrnn1(video)
        video_rnn2_out, _ = self.vrnn2(video_rnn1_out)
        utterance_video = video_rnn2_out                          # [T, B, 2Hv]

        v_pooled = self._pool_seq(utterance_video, self.kv, self.pool_visual)  # [B, kv, 2Hv]
        v_embeds = self.visual_proj(v_pooled)                    # [B, kv, D]
        self.v_embeds = v_embeds
        # ===== Video =====
        if video is not None and self.visual_proj is not None:
            prefix_tokens.append(v_embeds)
            
            
            
        audio_rnn1_out, _ = self.arnn1(acoustic)
        audio_rnn2_out, _ = self.arnn2(audio_rnn1_out)
        utterance_audio = audio_rnn2_out                         # [T, B, 2Ha]

        a_pooled = self._pool_seq(utterance_audio, self.ka, self.pool_acoustic)  # [B, ka, 2Ha]
        a_embeds = self.acoustic_proj(a_pooled)                  # [B, ka, D]
        self.a_embeds = a_embeds
        # ===== Audio =====
        if acoustic is not None and self.acoustic_proj is not None:  
            prefix_tokens.append(a_embeds)
        physio_rnn1_out, _ = self.prnn1(physio)
        physio_rnn2_out, _ = self.prnn2(physio_rnn1_out)
        utterance_physio = physio_rnn2_out                       # [T, B, 2Hp]

        p_pooled = self._pool_seq(utterance_physio, self.kp, self.pool_physio)  # [B, kp, 2Hp]
        p_embeds = self.physio_proj(p_pooled)                    # [B, kp, D]
        self.p_embeds = p_embeds
        # ===== Physio =====
        if physio is not None and self.physio_proj is not None:
            prefix_tokens.append(p_embeds)

        if len(prefix_tokens) == 0:
            raise ValueError("No modality prefix tokens available (check visual/acoustic/physio).")

        prefix_embeds = torch.cat(prefix_tokens, dim=1)  # [B, P, D]
        P = prefix_embeds.size(1)

        prefix_mask = torch.ones((B, P), dtype=attention_mask.dtype, device=device)
        attn_mask_cat = torch.cat([prefix_mask, attention_mask], dim=1)
    
    
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask_cat)
        hidden = outputs.last_hidden_state

        pooled = self._masked_mean(hidden, attn_mask_cat)
        logits = self.classifier(pooled)
        return logits