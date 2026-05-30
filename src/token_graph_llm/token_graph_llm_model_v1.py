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


class GraphCausalDecoderBlock(nn.Module):
    """Causal language decoder implemented as graph message passing."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.prefix_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.graph_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.update = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, generated: Tensor, graph_nodes: Tensor, *, node_mask: Tensor, node_bias: Tensor) -> Tensor:
        batch, seq_len, dim = generated.shape
        prefix_msg = torch.zeros_like(generated)
        if seq_len > 1:
            dst = generated.unsqueeze(2).expand(batch, seq_len, seq_len, dim)
            src = generated.unsqueeze(1).expand(batch, seq_len, seq_len, dim)
            scores = self.prefix_gate(torch.cat([dst, src], dim=-1)).squeeze(-1)
            causal = torch.tril(torch.ones(seq_len, seq_len, device=generated.device, dtype=torch.bool), diagonal=-1)
            scores = scores.masked_fill(~causal.unsqueeze(0), -1e4)
            weights = torch.softmax(scores, dim=-1) * causal.unsqueeze(0).to(dtype=generated.dtype)
            prefix_msg = torch.bmm(weights, generated)

        graph_scores = torch.einsum("btd,bnd->btn", generated, graph_nodes) / math.sqrt(dim)
        graph_scores = graph_scores + node_bias.unsqueeze(1)
        graph_scores = graph_scores.masked_fill(~node_mask.unsqueeze(1), -1e4)
        graph_weights = torch.softmax(graph_scores, dim=-1)
        graph_msg = torch.bmm(graph_weights, graph_nodes)
        delta = self.update(torch.cat([generated, prefix_msg, graph_msg], dim=-1))
        return self.norm(generated + self.dropout(delta))


class TokenGraphLanguageModel(nn.Module):
    """Graph-native language model.

    It predicts natural-language tokens from graph message passing only. There is
    no external LLM, Transformer attention, memory recall score, or token prior
    added to the final logits.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        node_type_count: int,
        edge_type_count: int,
        dim: int = 192,
        graph_layers: int = 4,
        decoder_layers: int = 4,
        max_sequence_tokens: int = 160,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = TokenGraphEncoder(vocab_size, node_type_count, edge_type_count, dim=dim, layers=graph_layers, dropout=dropout)
        self.sequence_pos_emb = nn.Embedding(max_sequence_tokens, dim)
        self.decoder_input_norm = nn.LayerNorm(dim)
        self.decoder_blocks = nn.ModuleList([GraphCausalDecoderBlock(dim, dropout) for _ in range(decoder_layers)])
        self.node_bias = nn.Linear(dim, 1)
        self.path_query = nn.Linear(dim, dim)
        self.path_key = nn.Linear(dim, dim)
        self.lm_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.encoder.token_emb.weight
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

    def encode_graph(self, batch: dict[str, Tensor]) -> Tensor:
        return self.encoder(
            batch["node_token_ids"],
            batch["node_types"],
            batch["node_mask"],
            batch["edge_src"],
            batch["edge_dst"],
            batch["edge_types"],
            batch["edge_mask"],
        )

    def decode_states(self, batch: dict[str, Tensor], graph_nodes: Tensor, token_prefix: Tensor) -> tuple[Tensor, Tensor]:
        node_mask = batch["node_mask"]
        node_bias = self.node_bias(graph_nodes).squeeze(-1).masked_fill(~node_mask, -1e4)
        positions = torch.arange(token_prefix.size(1), device=token_prefix.device).unsqueeze(0).expand(token_prefix.size(0), -1)
        positions = positions.clamp_max(self.sequence_pos_emb.num_embeddings - 1)
        generated = self.decoder_input_norm(self.encoder.token_emb(token_prefix) + self.sequence_pos_emb(positions))
        for block in self.decoder_blocks:
            generated = block(generated, graph_nodes, node_mask=node_mask, node_bias=node_bias)
        path_logits = torch.einsum("btd,bnd->btn", self.path_query(generated), self.path_key(graph_nodes)) / math.sqrt(graph_nodes.size(-1))
        path_logits = path_logits.masked_fill(~node_mask.unsqueeze(1), -1e4)
        return generated, path_logits

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        graph_nodes = self.encode_graph(batch)
        token_in = batch["sequence_ids"][:, :-1]
        states, path_logits = self.decode_states(batch, graph_nodes, token_in)
        logits = self.lm_head(self.lm_norm(states))
        return {"logits": logits, "path_logits": path_logits}

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
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> Tensor:
        self.eval()
        graph_nodes = self.encode_graph(batch)
        ids = torch.full((graph_nodes.size(0), 1), bos_id, dtype=torch.long, device=graph_nodes.device)
        done = torch.zeros(graph_nodes.size(0), dtype=torch.bool, device=graph_nodes.device)
        for step in range(max_new_tokens):
            states, _ = self.decode_states(batch, graph_nodes, ids)
            logits = self.lm_head(self.lm_norm(states[:, -1]))
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


def token_graph_lm_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    *,
    pad_id: int,
    eos_id: int | None = None,
    lm_weight: float = 1.0,
    token_path_weight: float = 0.0,
    transition_path_weight: float = 0.0,
    relation_transition_weight: float = 0.0,
    causal_path_weight: float = 0.0,
    non_eos_weight: float = 0.0,
    non_eos_steps: int = 0,
    label_smoothing: float = 0.0,
) -> dict[str, Tensor]:
    target = batch["sequence_ids"][:, 1:]
    previous = batch["sequence_ids"][:, :-1]
    lm_loss = F.cross_entropy(
        outputs["logits"].reshape(-1, outputs["logits"].size(-1)),
        target.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=float(label_smoothing),
    )
    non_eos_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    if non_eos_weight and non_eos_weight > 0.0 and eos_id is not None and non_eos_steps > 0:
        early_width = min(int(non_eos_steps), target.size(1))
        early_target = target[:, :early_width]
        early_logits = outputs["logits"][:, :early_width]
        early_mask = early_target.ne(pad_id) & early_target.ne(int(eos_id))
        if bool(early_mask.any()):
            eos_probs = F.softmax(early_logits, dim=-1)[..., int(eos_id)]
            non_eos_loss = -torch.log((1.0 - eos_probs).clamp_min(1e-8))[early_mask].mean()
    token_path_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    if token_path_weight and token_path_weight > 0.0:
        token_match = batch["node_token_ids"].unsqueeze(1).eq(target.unsqueeze(-1))
        positive = token_match & batch["node_mask"].unsqueeze(1) & target.ne(pad_id).unsqueeze(-1)
        has_positive = positive.any(dim=-1)
        if bool(has_positive.any()):
            log_probs = F.log_softmax(outputs["path_logits"], dim=-1)
            positive_log_probs = log_probs.masked_fill(~positive, -1e9)
            token_path_loss = -torch.logsumexp(positive_log_probs, dim=-1)[has_positive].mean()

    transition_path_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    transition_positive = None
    if transition_path_weight and transition_path_weight > 0.0:
        batch_size, seq_len = target.shape
        node_count = batch["node_token_ids"].size(1)
        transition_positive = torch.zeros(batch_size, seq_len, node_count, dtype=torch.bool, device=target.device)
        node_ids = batch["node_token_ids"]
        for batch_index in range(batch_size):
            valid_edges = batch["edge_mask"][batch_index]
            if not bool(valid_edges.any()):
                continue
            src = batch["edge_src"][batch_index, valid_edges]
            dst = batch["edge_dst"][batch_index, valid_edges]
            src_token_ids = node_ids[batch_index, src]
            dst_token_ids = node_ids[batch_index, dst]
            for token_index in range(seq_len):
                if int(target[batch_index, token_index].detach().cpu()) == pad_id:
                    continue
                edge_hits = (src_token_ids == previous[batch_index, token_index]) & (dst_token_ids == target[batch_index, token_index])
                if bool(edge_hits.any()):
                    transition_positive[batch_index, token_index, dst[edge_hits]] = True
        transition_positive = transition_positive & batch["node_mask"].unsqueeze(1)
        has_transition = transition_positive.any(dim=-1)
        if bool(has_transition.any()):
            log_probs = F.log_softmax(outputs["path_logits"], dim=-1)
            positive_log_probs = log_probs.masked_fill(~transition_positive, -1e9)
            transition_path_loss = -torch.logsumexp(positive_log_probs, dim=-1)[has_transition].mean()

    relation_transition_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    if relation_transition_weight and relation_transition_weight > 0.0:
        batch_size, seq_len = target.shape
        node_count = batch["node_token_ids"].size(1)
        relation_positive = torch.zeros(batch_size, seq_len, node_count, dtype=torch.bool, device=target.device)
        node_ids = batch["node_token_ids"]
        for batch_index in range(batch_size):
            valid_edges = batch["edge_mask"][batch_index]
            if not bool(valid_edges.any()):
                continue
            src = batch["edge_src"][batch_index, valid_edges]
            dst = batch["edge_dst"][batch_index, valid_edges]
            src_token_ids = node_ids[batch_index, src]
            dst_token_ids = node_ids[batch_index, dst]
            for token_index in range(seq_len):
                token_id = target[batch_index, token_index]
                if int(token_id.detach().cpu()) == pad_id:
                    continue
                direct_hits = dst_token_ids == token_id
                if bool(direct_hits.any()):
                    relation_positive[batch_index, token_index, dst[direct_hits]] = True
                    continue
                previous_hits = src_token_ids == previous[batch_index, token_index]
                if bool(previous_hits.any()):
                    relation_positive[batch_index, token_index, dst[previous_hits]] = True
        relation_positive = relation_positive & batch["node_mask"].unsqueeze(1)
        has_relation = relation_positive.any(dim=-1)
        if bool(has_relation.any()):
            log_probs = F.log_softmax(outputs["path_logits"], dim=-1)
            positive_log_probs = log_probs.masked_fill(~relation_positive, -1e9)
            relation_transition_loss = -torch.logsumexp(positive_log_probs, dim=-1)[has_relation].mean()

    causal_path_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
    if causal_path_weight and causal_path_weight > 0.0:
        probs = F.softmax(outputs["path_logits"], dim=-1)
        batch_size, seq_len, _ = probs.shape
        causal_terms: list[Tensor] = []
        for batch_index in range(batch_size):
            valid_edges = batch["edge_mask"][batch_index]
            if not bool(valid_edges.any()):
                continue
            src = batch["edge_src"][batch_index, valid_edges]
            dst = batch["edge_dst"][batch_index, valid_edges]
            for token_index in range(1, seq_len):
                if int(target[batch_index, token_index].detach().cpu()) == pad_id:
                    continue
                edge_mass = probs[batch_index, token_index - 1, src] * probs[batch_index, token_index, dst]
                if edge_mass.numel():
                    causal_terms.append(-torch.log(edge_mass.sum().clamp_min(1e-8)))
        if causal_terms:
            causal_path_loss = torch.stack(causal_terms).mean()

    total = (
        lm_weight * lm_loss
        + token_path_weight * token_path_loss
        + transition_path_weight * transition_path_loss
        + relation_transition_weight * relation_transition_loss
        + causal_path_weight * causal_path_loss
        + non_eos_weight * non_eos_loss
    )
    return {
        "loss": total,
        "lm_loss": lm_loss.detach(),
        "non_eos_loss": non_eos_loss.detach(),
        "token_path_loss": token_path_loss.detach(),
        "transition_path_loss": transition_path_loss.detach(),
        "relation_transition_loss": relation_transition_loss.detach(),
        "causal_path_loss": causal_path_loss.detach(),
    }
