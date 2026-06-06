from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TokenGraphEncoder(nn.Module):
    def __init__(self, vocab_size: int, node_type_count: int, edge_type_count: int, *, dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.node_type_emb = nn.Embedding(node_type_count, dim)
        self.edge_type_emb = nn.Embedding(edge_type_count, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "msg": nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)),
                    "upd": nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)),
                    "norm": nn.LayerNorm(dim),
                }
            )
            for _ in range(layers)
        )

    def forward(self, token_ids: Tensor, node_types: Tensor, node_mask: Tensor, edge_src: Tensor, edge_dst: Tensor, edge_type: Tensor, edge_mask: Tensor) -> Tensor:
        h = self.input_norm(self.token_emb(token_ids) + self.node_type_emb(node_types))
        h = h * node_mask.unsqueeze(-1).to(dtype=h.dtype)
        for layer in self.layers:
            src_h = torch.gather(h, 1, edge_src.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            edge_h = self.edge_type_emb(edge_type)
            msg = layer["msg"](torch.cat([src_h, edge_h], dim=-1)) * edge_mask.unsqueeze(-1).to(dtype=h.dtype)
            agg = torch.zeros_like(h)
            for batch_index in range(h.size(0)):
                valid = edge_mask[batch_index]
                if bool(valid.any()):
                    agg[batch_index].index_add_(0, edge_dst[batch_index, valid], msg[batch_index, valid])
            h = layer["norm"](h + layer["upd"](torch.cat([h, agg], dim=-1)))
            h = h * node_mask.unsqueeze(-1).to(dtype=h.dtype)
        return h


class GraphAutoregressiveDecoderBlock(nn.Module):
    """Graph-native autoregressive decoder block.

    Generated tokens are treated as answer graph nodes. Each block propagates
    messages from earlier answer nodes and from scored context graph nodes. It
    intentionally avoids Transformer self-attention/cross-attention.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.prefix_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.context_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.update = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, generated: Tensor, context: Tensor, *, context_mask: Tensor, context_prior: Tensor) -> Tensor:
        batch, seq_len, dim = generated.shape
        prefix_msg = torch.zeros_like(generated)
        if seq_len > 1:
            dst = generated.unsqueeze(2).expand(batch, seq_len, seq_len, dim)
            src = generated.unsqueeze(1).expand(batch, seq_len, seq_len, dim)
            pair = torch.cat([dst, src], dim=-1)
            scores = self.prefix_gate(pair).squeeze(-1)
            causal = torch.tril(torch.ones(seq_len, seq_len, device=generated.device, dtype=torch.bool), diagonal=-1)
            scores = scores.masked_fill(~causal.unsqueeze(0), -1e4)
            weights = torch.softmax(scores, dim=-1) * causal.unsqueeze(0).to(dtype=generated.dtype)
            prefix_msg = torch.bmm(weights, generated)

        context_scores = torch.einsum("btd,bnd->btn", generated, context) / math.sqrt(dim)
        context_scores = context_scores + context_prior.unsqueeze(1)
        context_scores = context_scores.masked_fill(~context_mask.unsqueeze(1), -1e4)
        context_weights = torch.softmax(context_scores, dim=-1)
        context_msg = torch.bmm(context_weights, context)

        delta = self.update(torch.cat([generated, prefix_msg, context_msg], dim=-1))
        return self.norm(generated + self.dropout(delta))


class TokenGraphCausalLM(nn.Module):
    """Native graph language model.

    The context is a token graph. The generated answer prefix is treated as a
    causal token graph inside the decoder, not as an external LLM prompt.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        node_type_count: int,
        edge_type_count: int,
        dim: int = 256,
        graph_layers: int = 4,
        decoder_layers: int = 4,
        heads: int = 4,
        max_answer_tokens: int = 192,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        graph_prior_init: float = 0.35,
        graph_prior_max: float = 1.5,
    ) -> None:
        super().__init__()
        self.graph_prior_max = float(graph_prior_max)
        self.encoder = TokenGraphEncoder(vocab_size, node_type_count, edge_type_count, dim=dim, layers=graph_layers, dropout=dropout)
        self.answer_pos_emb = nn.Embedding(max_answer_tokens, dim)
        self.decoder_input_norm = nn.LayerNorm(dim)
        self.decoder_blocks = nn.ModuleList([GraphAutoregressiveDecoderBlock(dim, dropout) for _ in range(decoder_layers)])
        self.lm_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.encoder.token_emb.weight
        self.context_token_score = nn.Linear(dim, 1)
        self.answer_overlap_score = nn.Linear(dim, 1)
        self.tunnel_query = nn.Linear(dim, dim)
        self.tunnel_key = nn.Linear(dim, dim)
        self.graph_prior_scale = nn.Parameter(torch.tensor(float(graph_prior_init)))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def encode_context(self, batch: dict[str, Tensor]) -> Tensor:
        return self.encoder(
            batch["node_token_ids"],
            batch["node_types"],
            batch["node_mask"],
            batch["edge_src"],
            batch["edge_dst"],
            batch["edge_types"],
            batch["edge_mask"],
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        context = self.encode_context(batch)
        context_mask = batch["node_mask"]
        context_token_score = self.context_token_score(context).squeeze(-1)
        answer_overlap_score = self.answer_overlap_score(context).squeeze(-1)
        context_prior = context_token_score + answer_overlap_score
        context_prior = context_prior.masked_fill(~context_mask, -1e4)
        answer_in = batch["answer_ids"][:, :-1]
        positions = torch.arange(answer_in.size(1), device=answer_in.device).unsqueeze(0).expand(answer_in.size(0), -1)
        generated = self.decoder_input_norm(self.encoder.token_emb(answer_in) + self.answer_pos_emb(positions))
        for block in self.decoder_blocks:
            generated = block(generated, context, context_mask=context_mask, context_prior=context_prior)
        tunnel_logits = torch.einsum(
            "btd,bnd->btn",
            self.tunnel_query(generated),
            self.tunnel_key(context),
        ) / math.sqrt(context.size(-1))
        tunnel_logits = tunnel_logits.masked_fill(~context_mask.unsqueeze(1), -1e4)
        graph_weights = torch.softmax(tunnel_logits + context_prior.unsqueeze(1), dim=-1)
        graph_vocab_prior = torch.zeros(
            generated.size(0),
            generated.size(1),
            self.lm_head.out_features,
            dtype=generated.dtype,
            device=generated.device,
        )
        graph_vocab_prior.scatter_add_(
            2,
            batch["node_token_ids"].unsqueeze(1).expand(-1, generated.size(1), -1),
            graph_weights,
        )
        graph_state_logits = self.lm_head(self.lm_norm(generated))
        graph_prior_logits = torch.zeros_like(graph_state_logits)
        logits = graph_state_logits
        return {
            "logits": logits,
            "graph_state_logits": graph_state_logits,
            "graph_prior_logits": graph_prior_logits,
            "context_token_score": context_token_score,
            "answer_overlap_score": answer_overlap_score,
            "tunnel_logits": tunnel_logits,
            "context_prior": context_prior,
        }

    @torch.no_grad()
    def generate(
        self,
        batch: dict[str, Tensor],
        *,
        bos_id: int,
        eos_id: int,
        pad_id: int = 0,
        unk_id: int | None = None,
        max_new_tokens: int,
        min_new_tokens: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        base_top_k: int = 0,
        graph_vocab_top_k: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> Tensor:
        self.eval()
        context = self.encode_context(batch)
        context_mask = batch["node_mask"]
        context_token_score = self.context_token_score(context).squeeze(-1)
        answer_overlap_score = self.answer_overlap_score(context).squeeze(-1)
        context_prior = (context_token_score + answer_overlap_score).masked_fill(~context_mask, -1e4)
        ids = torch.full((context.size(0), 1), bos_id, dtype=torch.long, device=context.device)
        done = torch.zeros(context.size(0), dtype=torch.bool, device=context.device)
        for step in range(max_new_tokens):
            positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand(ids.size(0), -1)
            generated = self.decoder_input_norm(self.encoder.token_emb(ids) + self.answer_pos_emb(positions.clamp_max(self.answer_pos_emb.num_embeddings - 1)))
            for block in self.decoder_blocks:
                generated = block(generated, context, context_mask=context_mask, context_prior=context_prior)
            base_logits = self.lm_head(self.lm_norm(generated[:, -1]))
            logits = base_logits
            logits[:, pad_id] = -1e9
            logits[:, bos_id] = -1e9
            if unk_id is not None:
                logits[:, int(unk_id)] = -1e9
            if step < min_new_tokens:
                logits[:, eos_id] = -1e9
            if base_top_k and base_top_k > 0:
                allowed = torch.zeros_like(logits, dtype=torch.bool)
                k = min(int(base_top_k), base_logits.size(-1))
                allowed.scatter_(1, torch.topk(base_logits, k=k, dim=-1).indices, True)
                allowed[:, pad_id] = False
                allowed[:, bos_id] = False
                if unk_id is not None:
                    allowed[:, int(unk_id)] = False
                if step < min_new_tokens:
                    allowed[:, eos_id] = False
                logits = logits.masked_fill(~allowed, -1e9)
            if repetition_penalty and repetition_penalty > 1.0:
                for batch_index in range(ids.size(0)):
                    for token_id in set(int(x) for x in ids[batch_index].tolist()):
                        if token_id in {pad_id, bos_id}:
                            continue
                        value = logits[batch_index, token_id]
                        logits[batch_index, token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
            if no_repeat_ngram_size and ids.size(1) >= no_repeat_ngram_size:
                n = int(no_repeat_ngram_size)
                for batch_index in range(ids.size(0)):
                    prefix = tuple(int(x) for x in ids[batch_index, -(n - 1) :].tolist())
                    banned: set[int] = set()
                    row_ids = [int(x) for x in ids[batch_index].tolist()]
                    for pos in range(0, len(row_ids) - n + 1):
                        if tuple(row_ids[pos : pos + n - 1]) == prefix:
                            banned.add(row_ids[pos + n - 1])
                    if banned:
                        logits[batch_index, list(banned)] = -1e9
            if temperature and temperature != 1.0:
                logits = logits / max(float(temperature), 1e-5)
            if top_k and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                threshold = torch.topk(logits, k=k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, -1e9)
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_id = torch.argmax(logits, dim=-1)
            next_id = torch.where(done, torch.full_like(next_id, eos_id), next_id)
            ids = torch.cat([ids, next_id[:, None]], dim=1)
            done = done | (next_id == eos_id)
            if bool(done.all()):
                break
        return ids[:, 1:]


def graph_lm_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    pad_id: int,
    lm_weight: float = 1.0,
    graph_state_weight: float = 0.0,
    support_weight: float = 0.15,
    overlap_weight: float = 0.08,
    tunnel_weight: float = 0.08,
    next_token_node_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> dict[str, Tensor]:
    target = batch["answer_ids"][:, 1:]
    lm_loss = F.cross_entropy(
        outputs["logits"].reshape(-1, outputs["logits"].size(-1)),
        target.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=float(label_smoothing),
    )
    graph_state_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    if graph_state_weight and "graph_state_logits" in outputs:
        graph_state_loss = F.cross_entropy(
            outputs["graph_state_logits"].reshape(-1, outputs["graph_state_logits"].size(-1)),
            target.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=float(label_smoothing),
        )
    node_mask = batch["node_mask"].to(dtype=outputs["context_token_score"].dtype)
    support_loss = F.binary_cross_entropy_with_logits(outputs["context_token_score"], batch["support_labels"], reduction="none")
    support_loss = (support_loss * node_mask).sum() / node_mask.sum().clamp_min(1.0)
    overlap_loss = F.binary_cross_entropy_with_logits(outputs["answer_overlap_score"], batch["answer_overlap_labels"], reduction="none")
    overlap_loss = (overlap_loss * node_mask).sum() / node_mask.sum().clamp_min(1.0)

    support_targets = batch["support_labels"].unsqueeze(1).expand_as(outputs["tunnel_logits"])
    tunnel_mask = node_mask.unsqueeze(1).expand_as(outputs["tunnel_logits"])
    tunnel_loss = F.binary_cross_entropy_with_logits(outputs["tunnel_logits"], support_targets, reduction="none")
    tunnel_loss = (tunnel_loss * tunnel_mask).sum() / tunnel_mask.sum().clamp_min(1.0)
    next_token_node_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    if next_token_node_weight and next_token_node_weight > 0.0:
        node_positive = (
            (batch["support_labels"] > 0.5)
            | (batch["answer_overlap_labels"] > 0.5)
        ) & batch["node_mask"]
        token_match = batch["node_token_ids"].unsqueeze(1).eq(target.unsqueeze(-1))
        target_valid = target.ne(pad_id).unsqueeze(-1)
        positive = token_match & node_positive.unsqueeze(1) & target_valid
        has_positive = positive.any(dim=-1)
        if bool(has_positive.any()):
            scored_tunnel = outputs["tunnel_logits"] + outputs["context_prior"].unsqueeze(1)
            log_probs = F.log_softmax(scored_tunnel, dim=-1)
            positive_log_probs = log_probs.masked_fill(~positive, -1e9)
            per_token = -torch.logsumexp(positive_log_probs, dim=-1)
            next_token_node_loss = per_token[has_positive].mean()
    total = (
        lm_weight * lm_loss
        + graph_state_weight * graph_state_loss
        + support_weight * support_loss
        + overlap_weight * overlap_loss
        + tunnel_weight * tunnel_loss
        + next_token_node_weight * next_token_node_loss
    )
    return {
        "loss": total,
        "lm_loss": lm_loss.detach(),
        "graph_state_loss": graph_state_loss.detach(),
        "support_loss": support_loss.detach(),
        "overlap_loss": overlap_loss.detach(),
        "tunnel_loss": tunnel_loss.detach(),
        "next_token_node_loss": next_token_node_loss.detach(),
    }
