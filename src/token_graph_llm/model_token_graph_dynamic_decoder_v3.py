from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LearnedEdgeGraphLayer(nn.Module):
    """Message-passing layer with learned edge activation.

    The dataset supplies candidate token edges and an explicit edge type id.
    This layer decides how much each candidate edge should participate through
    a learned gate; fixed candidate edges are therefore not fixed evidence.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.message = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.update = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, edge_h: Tensor, edge_src: Tensor, edge_dst: Tensor, edge_mask: Tensor) -> tuple[Tensor, Tensor]:
        src_h = torch.gather(h, 1, edge_src.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        dst_h = torch.gather(h, 1, edge_dst.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        gate_logits = self.edge_gate(torch.cat([src_h, dst_h, edge_h], dim=-1)).squeeze(-1)
        gate = torch.sigmoid(gate_logits) * edge_mask.to(dtype=h.dtype)
        msg = self.message(torch.cat([src_h, edge_h], dim=-1)) * gate.unsqueeze(-1)
        agg = torch.zeros_like(h)
        degree = torch.zeros(h.size(0), h.size(1), dtype=h.dtype, device=h.device)
        safe_dst = edge_dst.clamp(0, h.size(1) - 1)
        agg.scatter_add_(1, safe_dst.unsqueeze(-1).expand(-1, -1, h.size(-1)), msg)
        degree.scatter_add_(1, safe_dst, gate)
        agg = agg / degree.clamp_min(1.0).unsqueeze(-1)
        return self.norm(h + self.dropout(self.update(torch.cat([h, agg], dim=-1)))), gate_logits


class TokenGraphEncoderV3(nn.Module):
    def __init__(self, vocab_size: int, node_type_count: int, edge_type_count: int, *, dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.node_type_emb = nn.Embedding(node_type_count, dim)
        self.edge_type_emb = nn.Embedding(edge_type_count, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([LearnedEdgeGraphLayer(dim, dropout) for _ in range(layers)])
        self.edge_type_head = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, edge_type_count))

    def forward(
        self,
        token_ids: Tensor,
        node_types: Tensor,
        node_mask: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_type: Tensor,
        edge_mask: Tensor,
    ) -> dict[str, Tensor]:
        h = self.input_norm(self.token_emb(token_ids) + self.node_type_emb(node_types))
        h = h * node_mask.unsqueeze(-1).to(dtype=h.dtype)
        edge_h = self.edge_type_emb(edge_type)
        edge_gate_logits = torch.zeros(edge_type.shape, dtype=h.dtype, device=h.device)
        for layer in self.layers:
            h, edge_gate_logits = layer(h, edge_h, edge_src, edge_dst, edge_mask)
            h = h * node_mask.unsqueeze(-1).to(dtype=h.dtype)
        src_h = torch.gather(h, 1, edge_src.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        dst_h = torch.gather(h, 1, edge_dst.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        edge_type_logits = self.edge_type_head(torch.cat([src_h, dst_h], dim=-1))
        return {"node_states": h, "edge_gate_logits": edge_gate_logits, "edge_type_logits": edge_type_logits}


class DynamicTokenGraphGeneratorBlock(nn.Module):
    """Causal graph decoder block where generated tokens are graph nodes."""

    def __init__(self, dim: int, dropout: float, *, prefix_window: int = 64) -> None:
        super().__init__()
        self.prefix_window = max(1, int(prefix_window))
        self.answer_edge_emb = nn.Embedding(2, dim)
        self.prefix_gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.context_gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.update = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, answer_nodes: Tensor, context: Tensor, *, context_mask: Tensor, context_prior: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq_len, dim = answer_nodes.shape
        prefix_msg = torch.zeros_like(answer_nodes)
        if seq_len > 1:
            window = min(self.prefix_window, seq_len - 1)
            positions = torch.arange(seq_len, device=answer_nodes.device)
            offsets = torch.arange(1, window + 1, device=answer_nodes.device)
            src_pos = positions.unsqueeze(1) - offsets.unsqueeze(0)
            valid_prefix = src_pos >= 0
            safe_src_pos = src_pos.clamp_min(0)
            source_bank = answer_nodes.unsqueeze(1).expand(-1, seq_len, -1, -1)
            gather_index = safe_src_pos.view(1, seq_len, window, 1).expand(batch, -1, -1, dim)
            src = torch.gather(source_bank, 2, gather_index)
            dst = answer_nodes.unsqueeze(2).expand(-1, -1, window, -1)
            rel = self.answer_edge_emb.weight[0].view(1, 1, 1, dim).expand(batch, seq_len, window, -1)
            scores = self.prefix_gate(torch.cat([src, dst, rel], dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~valid_prefix.view(1, seq_len, window), -1e4)
            weights = torch.softmax(scores, dim=-1) * valid_prefix.view(1, seq_len, window).to(dtype=answer_nodes.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            prefix_msg = (weights.unsqueeze(-1) * src).sum(dim=2)

        dst = answer_nodes.unsqueeze(2).expand(batch, seq_len, context.size(1), dim)
        src = context.unsqueeze(1).expand(batch, seq_len, context.size(1), dim)
        rel = self.answer_edge_emb.weight[1].view(1, 1, 1, dim).expand_as(src)
        context_scores = self.context_gate(torch.cat([src, dst, rel], dim=-1)).squeeze(-1)
        context_scores = context_scores + context_prior.unsqueeze(1)
        context_scores = context_scores.masked_fill(~context_mask.unsqueeze(1), -1e4)
        context_weights = torch.softmax(context_scores, dim=-1)
        context_msg = torch.bmm(context_weights, context)
        delta = self.update(torch.cat([answer_nodes, prefix_msg, context_msg], dim=-1))
        return self.norm(answer_nodes + self.dropout(delta)), context_scores


class TokenGraphDynamicCausalLMV3(nn.Module):
    """Token-level graph-native causal language model.

    No Transformer attention blocks are used. Prompt/context tokens and generated
    tokens are represented as graph nodes. Candidate edge types are explicit,
    while edge activation is learned by gates inside graph propagation.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        node_type_count: int,
        edge_type_count: int,
        dim: int = 384,
        graph_layers: int = 6,
        decoder_layers: int = 8,
        heads: int = 4,
        max_answer_tokens: int = 192,
        dropout: float = 0.1,
        tie_embeddings: bool = False,
        graph_prior_init: float = 0.0,
        graph_prior_max: float = 1.0,
        prefix_window: int = 64,
    ) -> None:
        super().__init__()
        self.graph_prior_max = float(graph_prior_max)
        self.encoder = TokenGraphEncoderV3(vocab_size, node_type_count, edge_type_count, dim=dim, layers=graph_layers, dropout=dropout)
        self.answer_node_type_emb = nn.Embedding(1, dim)
        self.answer_pos_emb = nn.Embedding(max_answer_tokens, dim)
        self.decoder_input_norm = nn.LayerNorm(dim)
        self.decoder_blocks = nn.ModuleList([DynamicTokenGraphGeneratorBlock(dim, dropout, prefix_window=prefix_window) for _ in range(decoder_layers)])
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

    def encode_context(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.encoder(
            batch["node_token_ids"],
            batch["node_types"],
            batch["node_mask"],
            batch["edge_src"],
            batch["edge_dst"],
            batch["edge_types"],
            batch["edge_mask"],
        )

    def _decode_answer_nodes(self, answer_in: Tensor, context: Tensor, context_mask: Tensor, context_prior: Tensor) -> tuple[Tensor, Tensor]:
        positions = torch.arange(answer_in.size(1), device=answer_in.device).unsqueeze(0).expand(answer_in.size(0), -1)
        node_type = self.answer_node_type_emb.weight[0].view(1, 1, -1)
        generated = self.decoder_input_norm(
            self.encoder.token_emb(answer_in)
            + self.answer_pos_emb(positions.clamp_max(self.answer_pos_emb.num_embeddings - 1))
            + node_type
        )
        tunnel_logits = torch.zeros(answer_in.size(0), answer_in.size(1), context.size(1), dtype=context.dtype, device=context.device)
        for block in self.decoder_blocks:
            generated, tunnel_logits = block(generated, context, context_mask=context_mask, context_prior=context_prior)
        return generated, tunnel_logits

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        encoded = self.encode_context(batch)
        context = encoded["node_states"]
        context_mask = batch["node_mask"]
        context_token_score = self.context_token_score(context).squeeze(-1)
        answer_overlap_score = self.answer_overlap_score(context).squeeze(-1)
        context_prior = (context_token_score + answer_overlap_score).masked_fill(~context_mask, -1e4)
        answer_in = batch["answer_ids"][:, :-1]
        generated, decoder_tunnel_logits = self._decode_answer_nodes(answer_in, context, context_mask, context_prior)
        tunnel_logits = torch.einsum("btd,bnd->btn", self.tunnel_query(generated), self.tunnel_key(context)) / math.sqrt(context.size(-1))
        tunnel_logits = tunnel_logits + decoder_tunnel_logits
        tunnel_logits = tunnel_logits.masked_fill(~context_mask.unsqueeze(1), -1e4)
        logits = self.lm_head(self.lm_norm(generated))
        return {
            "logits": logits,
            "graph_state_logits": logits,
            "context_token_score": context_token_score,
            "answer_overlap_score": answer_overlap_score,
            "tunnel_logits": tunnel_logits,
            "context_prior": context_prior,
            "edge_gate_logits": encoded["edge_gate_logits"],
            "edge_type_logits": encoded["edge_type_logits"],
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
        encoded = self.encode_context(batch)
        context = encoded["node_states"]
        context_mask = batch["node_mask"]
        context_prior = (self.context_token_score(context).squeeze(-1) + self.answer_overlap_score(context).squeeze(-1)).masked_fill(~context_mask, -1e4)
        ids = torch.full((context.size(0), 1), bos_id, dtype=torch.long, device=context.device)
        done = torch.zeros(context.size(0), dtype=torch.bool, device=context.device)
        for step in range(max_new_tokens):
            generated, _ = self._decode_answer_nodes(ids, context, context_mask, context_prior)
            logits = self.lm_head(self.lm_norm(generated[:, -1]))
            logits[:, pad_id] = -1e9
            logits[:, bos_id] = -1e9
            if unk_id is not None:
                logits[:, int(unk_id)] = -1e9
            if step < min_new_tokens:
                logits[:, eos_id] = -1e9
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
                next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
            else:
                next_id = torch.argmax(logits, dim=-1)
            next_id = torch.where(done, torch.full_like(next_id, eos_id), next_id)
            ids = torch.cat([ids, next_id[:, None]], dim=1)
            done = done | (next_id == eos_id)
            if bool(done.all()):
                break
        return ids[:, 1:]


def graph_lm_loss_v3(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    pad_id: int,
    lm_weight: float = 1.0,
    graph_state_weight: float = 0.0,
    support_weight: float = 0.12,
    overlap_weight: float = 0.05,
    tunnel_weight: float = 0.08,
    next_token_node_weight: float = 0.0,
    edge_type_weight: float = 0.05,
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
        node_positive = ((batch["support_labels"] > 0.5) | (batch["answer_overlap_labels"] > 0.5)) & batch["node_mask"]
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

    edge_type_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    if edge_type_weight and edge_type_weight > 0.0:
        edge_mask = batch["edge_mask"]
        if bool(edge_mask.any()):
            edge_type_loss = F.cross_entropy(
                outputs["edge_type_logits"][edge_mask],
                batch["edge_types"][edge_mask],
            )
    total = (
        lm_weight * lm_loss
        + graph_state_weight * graph_state_loss
        + support_weight * support_loss
        + overlap_weight * overlap_loss
        + tunnel_weight * tunnel_loss
        + next_token_node_weight * next_token_node_loss
        + edge_type_weight * edge_type_loss
    )
    return {
        "loss": total,
        "lm_loss": lm_loss.detach(),
        "graph_state_loss": graph_state_loss.detach(),
        "support_loss": support_loss.detach(),
        "overlap_loss": overlap_loss.detach(),
        "tunnel_loss": tunnel_loss.detach(),
        "next_token_node_loss": next_token_node_loss.detach(),
        "edge_type_loss": edge_type_loss.detach(),
    }
