''' Define the Transformer model '''
from multiprocessing import context
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from transformer.Layers import DecoderLayer_Flash
from transformer.utils import *
from transformer.Const import *

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, seq_lens=None, start_pos=0):
        if seq_lens is None:
            L = x.size(1)
            return x + self.pos_table[:, start_pos:start_pos+L].clone().detach()

        if x.dim() == 3:
             return x + self.pos_table[:, :x.size(1)].clone().detach().to(x.device)

        seq_lens = seq_lens.to(device=x.device, dtype=torch.long)
        total_seq_len = int(seq_lens.sum().item())
        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        token_seq_ids = torch.arange(seq_lens.size(0), device=x.device).repeat_interleave(seq_lens)
        seq_offsets = seq_starts[token_seq_ids]
        position_ids = torch.arange(total_seq_len, device=x.device) - seq_offsets
        pos_emb = self.pos_table[:, position_ids, :].squeeze(0)
        return x + pos_emb

class Decoder(nn.Module):
    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, flash_attn=True):

        super().__init__()
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        self.flash_attn = flash_attn
        self.layer_stack = nn.ModuleList([
            DecoderLayer_Flash(d_model, d_inner, n_head, d_k, dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, kv_cache=None):
        x = self.trg_word_emb(trg_seq)
        if kv_cache is not None:
            start_pos = trg_mask[0].item()
            x = self.position_enc(x, start_pos=start_pos)
        else:
            x = self.position_enc(x, seq_lens=trg_mask)
        x = self.dropout(x)
        for i, layer in enumerate(self.layer_stack):
            layer_kv = kv_cache[i] if kv_cache is not None else None
            x = layer(x, trg_mask, enc_output, src_mask, kv_cache=layer_kv)
        x = self.layer_norm(x)
        return x

from transformers import ModernBertModel, AutoTokenizer, AutoConfig # Change DebertaV2Model to ModernBertModel
from transformer.Const import *

class Seq2SeqModelWithFlashAttn(nn.Module):
    def __init__(
        self,
        transformer_model_path: str = "answerdotai/ModernBERT-base", # Default path
        freeze_encoder: bool = True,
        weight_dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        super().__init__()
        encoder_kwargs = {}
        if weight_dtype is not None:
            encoder_kwargs["torch_dtype"] = weight_dtype

        # *** IMPORTANT: Change here from DebertaV2Model to ModernBertModel ***
        self.encoder = ModernBertModel.from_pretrained(transformer_model_path, **encoder_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)

        config = AutoConfig.from_pretrained(transformer_model_path)
        hidden_size = config.hidden_size # 768 for base

        self.decoder = Decoder(
            n_trg_vocab=len(self.tokenizer),
            d_word_vec=hidden_size,
            n_layers=12,
            n_head=12,
            d_k=hidden_size // 12,
            d_v=hidden_size // 12,
            d_model=hidden_size,
            d_inner=hidden_size * 2, # ModernBert FFN expansion (was 4 for DeBERTa)
            pad_idx=self.tokenizer.pad_token_id,
            n_position=MAX_TARGET_LEN,
            dropout=0.1,
            flash_attn=True)

        self.output_projection = nn.Linear(hidden_size, len(self.tokenizer), bias=False)
        self._cast_modules_to_dtype(weight_dtype)
        # Tying weights for ModernBert
        with torch.no_grad():
            self.decoder.trg_word_emb.weight.copy_(
                self.encoder.embeddings.tok_embeddings.weight
            )
        self.output_projection.weight = self.decoder.trg_word_emb.weight

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.weight_dtype = weight_dtype

    def forward(self, src_input_ids, trg_input_ids, src_seq_len, trg_seq_len):
        bsz = src_seq_len.size(0)
        max_src_len = src_input_ids.size(1)
        src_mask = torch.arange(max_src_len, device=src_input_ids.device).expand(bsz, max_src_len) < src_seq_len.unsqueeze(1)
        src_mask = src_mask.long()

        enc_outputs = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_mask
        )
        enc_output_padded = enc_outputs.last_hidden_state

        flat_enc_output_list = []
        for i in range(bsz):
            l = src_seq_len[i].item()
            flat_enc_output_list.append(enc_output_padded[i, :l])
        enc_output = torch.cat(flat_enc_output_list)

        dec_output = self.decoder(
            trg_seq=trg_input_ids,
            trg_mask=trg_seq_len,
            enc_output=enc_output,
            src_mask=src_seq_len
        )
        logits = self.output_projection(dec_output)
        return logits

    def top_k_top_p_filtering(self, logits, top_k, top_p):
        if logits.dim() == 1: logits = logits.unsqueeze(0)
        filter_value = -float('Inf')
        vocab_size = logits.size(-1)
        if top_k > 0 and top_k < vocab_size:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        src_seq_len: torch.Tensor,
        generation_limit: int,
        sampling: bool = False,
        top_k: int = 10,
        top_p: float = 0.9,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        device = self.output_projection.weight.device
        src_seq_len = src_seq_len.to(device=device, dtype=torch.int32)
        bsz = src_seq_len.size(0)

        max_src_len = input_ids.size(1)
        attention_mask = torch.arange(max_src_len, device=device).expand(bsz, max_src_len) < src_seq_len.unsqueeze(1)
        attention_mask = attention_mask.long()

        enc_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_output_padded = enc_outputs["last_hidden_state"]

        flat_enc_output_list = []
        for i in range(bsz):
            l = src_seq_len[i].item()
            flat_enc_output_list.append(enc_output_padded[i, :l])
        enc_output = torch.cat(flat_enc_output_list)

        if num_beams > 1 and bsz == 1:
            init_seq = torch.full((1, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=device)
            kv_cache = []
            max_gen_len = generation_limit + 1
            for _ in range(len(self.decoder.layer_stack)):
                k = torch.zeros(num_beams, max_gen_len, 12, 64, dtype=self.weight_dtype or torch.float32, device=device)
                v = torch.zeros(num_beams, max_gen_len, 12, 64, dtype=self.weight_dtype or torch.float32, device=device)
                kv_cache.append((k, v))

            enc_output_beams = enc_output.repeat(num_beams, 1)
            src_seq_len_beams = src_seq_len.repeat(num_beams)
            beam_scores = torch.zeros(num_beams, device=device); beam_scores[1:] = -1e9
            sequences = torch.full((num_beams, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=device)
            finished_beams = torch.zeros(num_beams, dtype=torch.bool, device=device)
            cache_seqlens = torch.zeros(num_beams, dtype=torch.int32, device=device)

            for step in range(generation_limit):
                if finished_beams.all(): break
                current_input_ids = sequences[:, -1:]
                dec_output = self.decoder(
                    trg_seq=current_input_ids, trg_mask=cache_seqlens,
                    enc_output=enc_output_beams, src_mask=src_seq_len_beams, kv_cache=kv_cache)
                next_token_logits = self.output_projection(dec_output).squeeze(1)
                
                if repetition_penalty != 1.0:
                    for i in range(num_beams):
                        prev_tokens = set(sequences[i].tolist())
                        for token_id in prev_tokens:
                            if next_token_logits[i, token_id] < 0:
                                next_token_logits[i, token_id] *= repetition_penalty
                            else:
                                next_token_logits[i, token_id] /= repetition_penalty

                next_token_scores = F.log_softmax(next_token_logits, dim=-1)

                next_scores = beam_scores.unsqueeze(1) + next_token_scores

                for i in range(num_beams):
                    if finished_beams[i]:
                        next_scores[i, :] = -1e9
                        next_scores[i, self.tokenizer.pad_token_id] = beam_scores[i]

                next_scores_flat = next_scores.view(-1)
                topk_scores, topk_indices = next_scores_flat.topk(num_beams, dim=0)

                beam_indices = topk_indices // next_token_logits.size(-1)
                token_indices = topk_indices % next_token_logits.size(-1)

                for layer_idx in range(len(kv_cache)):
                    k, v = kv_cache[layer_idx]
                    k[:] = k[beam_indices]
                    v[:] = v[beam_indices]

                beam_scores = topk_scores
                sequences = torch.cat([sequences[beam_indices], token_indices.unsqueeze(1)], dim=1)

                for i in range(num_beams):
                    is_finished = finished_beams[beam_indices[i]].item() or (token_indices[i].item() == self.tokenizer.sep_token_id)
                    finished_beams[i] = is_finished

                cache_seqlens = cache_seqlens[beam_indices] + 1

            best_idx = beam_scores.argmax().item()
            best_seq = sequences[best_idx]
            tokens = best_seq.tolist()
            if self.tokenizer.sep_token_id in tokens: tokens = tokens[: tokens.index(self.tokenizer.sep_token_id)]
            return [self.tokenizer.decode(tokens, skip_special_tokens=True)]

        sequences = torch.full((bsz, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        kv_cache = []
        max_gen_len = generation_limit + 1
        for _ in range(len(self.decoder.layer_stack)):
            k = torch.zeros(bsz, max_gen_len, 12, 64, dtype=self.weight_dtype or torch.float32, device=device)
            v = torch.zeros(bsz, max_gen_len, 12, 64, dtype=self.weight_dtype or torch.float32, device=device)
            kv_cache.append((k, v))
        cache_seqlens = torch.zeros(bsz, dtype=torch.int32, device=device)

        for step in range(generation_limit):
            current_input_ids = sequences[:, -1:]
            dec_output = self.decoder(
                trg_seq=current_input_ids, trg_mask=cache_seqlens,
                enc_output=enc_output, src_mask=src_seq_len, kv_cache=kv_cache)
            next_token_logits = self.output_projection(dec_output).squeeze(1)
            
            if repetition_penalty != 1.0:
                for i in range(bsz):
                    prev_tokens = set(sequences[i].tolist())
                    for token_id in prev_tokens:
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            if finished.any():
                next_token_logits[finished] = -float("inf")
                next_token_logits[finished, self.tokenizer.pad_token_id] = 0.0
            if sampling:
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            cache_seqlens += 1
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            for idx in range(bsz):
                if finished[idx]: continue
                if next_token[idx].item() == self.tokenizer.sep_token_id: finished[idx] = True
            if bool(torch.all(finished)): break

        output_text = []
        for seq in sequences:
            tokens = seq.tolist()
            if self.tokenizer.sep_token_id in tokens: tokens = tokens[: tokens.index(self.tokenizer.sep_token_id)]
            output_text.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return output_text

    def _cast_modules_to_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if dtype is None: return
        self.encoder.to(dtype=dtype)
        self.decoder.to(dtype=dtype)
        self.output_projection.to(dtype=dtype)

    def _tie_decoder_embeddings(self) -> None:
        with torch.no_grad():
            self.decoder.trg_word_emb.weight.copy_(
                self.encoder.embeddings.tok_embeddings.weight
            )
        self.output_projection.weight = self.decoder.trg_word_emb.weight
