''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence

class MultiHeadSelfAttention_Flash(nn.Module):
    ''' Multi-Head self Attention module using PyTorch SDPA '''

    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_qkv = nn.Linear(d_model, 3 * n_head * d_qkv, bias=False)
        self.w_o = nn.Linear(n_head * d_qkv, d_model, bias=False)
        self.dropout_rate = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal

    def forward(self, x, seq_lens, kv_cache=None):
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x

        if kv_cache is not None:
            # Inference path with KV Cache (x is B, 1, D)
            bsz = x.size(0)
            qkv = self.w_qkv(x).view(bsz, 1, 3, self.n_head, self.d_qkv)
            
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2] # (B, 1, H, D)
            q = q.transpose(1, 2) # (B, H, 1, D)
            k = k.transpose(1, 2) # (B, H, 1, D)
            v = v.transpose(1, 2) # (B, H, 1, D)

            prev_k, prev_v = kv_cache
            
            current_pos = seq_lens[0].item() # Assume all beam items are at same pos
            
            # Update cache
            # prev_k shape: (B, MaxLen, H, D)
            prev_k[:, current_pos, :, :] = k.squeeze(2)
            prev_v[:, current_pos, :, :] = v.squeeze(2)
            
            k_curr = prev_k[:, :current_pos+1].permute(0, 2, 1, 3) # (B, H, L, D)
            v_curr = prev_v[:, :current_pos+1].permute(0, 2, 1, 3)
            
            output = F.scaled_dot_product_attention(q, k_curr, v_curr, dropout_p=0.0, is_causal=False)
            output = output.transpose(1, 2).reshape(bsz, 1, self.n_head * self.d_qkv)

        else:
            # Training path (x is flattened)
            x_list = torch.split(x, seq_lens.tolist())
            x_padded = pad_sequence(x_list, batch_first=True) # (B, L, D)
            
            qkv = self.w_qkv(x_padded)
            qkv = qkv.view(x_padded.size(0), x_padded.size(1), 3, self.n_head, self.d_qkv)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2) # (B, H, L, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_rate, is_causal=self.causal)
            
            output = output.transpose(1, 2).reshape(x_padded.size(0), x_padded.size(1), -1)
            
            # Flatten back
            out_list = []
            for i, l in enumerate(seq_lens):
                out_list.append(output[i, :l])
            output = torch.cat(out_list)

        output = self.w_o(output)
        output = self.dropout_layer(output)
        output += residual
        output = self.layer_norm(output)

        return output

class MultiHeadCrossAttention_Flash(nn.Module):
    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_q = nn.Linear(d_model, n_head * d_qkv, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * n_head * d_qkv, bias=False)
        self.w_o = nn.Linear(n_head * d_qkv, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal

    def forward(self, x_q, x_kv, seq_lens_q, seq_lens_kv, kv_cache=None):
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x_q
        is_inference = x_q.dim() == 3

        if is_inference:
            # Inference: Q is (B, 1, D)
            bsz = x_q.size(0)
            q = self.w_q(x_q).view(bsz, 1, self.n_head, self.d_qkv).transpose(1, 2) # (B, H, 1, D)
            
            # KV is flattened encoder output. Need to unflatten.
            kv_list = torch.split(x_kv, seq_lens_kv.tolist())
            kv_padded = pad_sequence(kv_list, batch_first=True) # (B, L_k, D)
            
            kv = self.w_kv(kv_padded).view(bsz, -1, 2, self.n_head, self.d_qkv)
            k = kv[:, :, 0].transpose(1, 2) # (B, H, L_k, D)
            v = kv[:, :, 1].transpose(1, 2)
            
            # Create padding mask for Encoder
            max_len_k = kv_padded.size(1)
            key_padding_mask = torch.arange(max_len_k, device=x_q.device).expand(bsz, max_len_k) >= seq_lens_kv.unsqueeze(1)
            attn_mask = ~key_padding_mask # True for valid tokens
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, L_k)
            
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate)
            output = output.transpose(1, 2).reshape(bsz, 1, self.n_head * self.d_qkv)

        else:
            # Training: Q, KV flattened
            q_list = torch.split(x_q, seq_lens_q.tolist())
            q_padded = pad_sequence(q_list, batch_first=True)
            q = self.w_q(q_padded).view(q_padded.size(0), -1, self.n_head, self.d_qkv).transpose(1, 2)
            
            kv_list = torch.split(x_kv, seq_lens_kv.tolist())
            kv_padded = pad_sequence(kv_list, batch_first=True)
            kv = self.w_kv(kv_padded).view(kv_padded.size(0), -1, 2, self.n_head, self.d_qkv)
            k = kv[:, :, 0].transpose(1, 2)
            v = kv[:, :, 1].transpose(1, 2)

            # Mask
            bsz, max_len_k = kv_padded.shape[:2]
            key_padding_mask = torch.arange(max_len_k, device=x_q.device).expand(bsz, max_len_k) >= seq_lens_kv.unsqueeze(1)
            attn_mask = ~key_padding_mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, L_k)
            
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate)
            
            output = output.transpose(1, 2).reshape(q_padded.size(0), q_padded.size(1), -1)
            
            # Flatten back
            out_list = []
            for i, l in enumerate(seq_lens_q):
                out_list.append(output[i, :l])
            output = torch.cat(out_list)

        output = self.w_o(output)
        output = self.dropout_layer(output)
        output += residual
        output = self.layer_norm(output)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
