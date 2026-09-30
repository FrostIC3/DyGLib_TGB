"""
Drop-in pieces for DyGLib_TGB (models/DyGFormer.py, utils/utils.py, train_link_prediction.py).

Status:
  - Section 1 logic was checked against the original count_nodes_appearances (numpy equivalence, 50 random cases).
  - Sections 2-4 are NOT run here (no torch/TGB data in my sandbox). Smoke-test on a small dataset first.
Apply ONE change at a time and log validation MRR after each, so you know what helped.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# 1. SPEED: vectorised co-occurrence counts
#    Replace NeighborCooccurrenceEncoder.count_nodes_appearances in models/DyGFormer.py.
#    The original loops over the batch in Python with np.unique + torch.apply_ on CPU.
#    Memory is O(B * L_src * L_dst) booleans, fine for L <= 512 at moderate batch sizes.
# ----------------------------------------------------------------------------------------------
def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
    s = torch.from_numpy(src_padded_nodes_neighbor_ids).to(self.device)  # (B, Ls)
    d = torch.from_numpy(dst_padded_nodes_neighbor_ids).to(self.device)  # (B, Ld)

    s_in_s = (s.unsqueeze(2) == s.unsqueeze(1)).sum(-1)  # (B, Ls): count of each src neighbour inside src sequence
    d_in_d = (d.unsqueeze(2) == d.unsqueeze(1)).sum(-1)  # (B, Ld)
    s_in_d = (s.unsqueeze(2) == d.unsqueeze(1)).sum(-1)  # (B, Ls): count of each src neighbour inside dst sequence
    d_in_s = (d.unsqueeze(2) == s.unsqueeze(1)).sum(-1)  # (B, Ld)

    src_app = torch.stack([s_in_s, s_in_d], dim=-1).float()  # (B, Ls, 2)
    dst_app = torch.stack([d_in_s, d_in_d], dim=-1).float()  # (B, Ld, 2)
    src_app[s == 0] = 0.0  # padded positions
    dst_app[d == 0] = 0.0
    # OPTIONAL (task E): counts can reach hundreds, so compress them:
    # src_app, dst_app = torch.log1p(src_app), torch.log1p(dst_app)
    return src_app, dst_app


# ----------------------------------------------------------------------------------------------
# 2. ARCHITECTURE FIX: padding mask + masked mean pooling
#    Original: sequences are padded to the batch max length, the Transformer gets no mask, and the
#    final torch.mean(dim=1) averages over padded patches too. So an embedding depends on how long the
#    longest sequence in the *batch* is (large train batches vs. batch size 1 on tgbl-wiki eval).
#
#    (a) TransformerEncoder.forward: add a key_padding_mask argument.
# ----------------------------------------------------------------------------------------------
def transformer_encoder_forward(self, inputs: torch.Tensor, key_padding_mask: torch.Tensor = None):
    """key_padding_mask: bool Tensor (batch_size, num_patches), True = padded token (ignored by attention)."""
    x = inputs.transpose(0, 1)
    x = self.norm_layers[0](x)
    h = self.multi_head_attention(query=x, key=x, value=x, key_padding_mask=key_padding_mask, need_weights=False)[0].transpose(0, 1)
    outputs = inputs + self.dropout(h)
    h = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
    return outputs + self.dropout(h)


#    (b) In DyGFormer.compute_src_dst_node_temporal_embeddings, right after the two get_patches calls
#        (a patch is padding only if every id inside it is 0; position 0 is the target node, never padding):
def build_patch_pad_mask(src_padded_ids: np.ndarray, dst_padded_ids: np.ndarray, patch_size: int, device):
    def one(ids):
        b, l = ids.shape
        return torch.from_numpy(ids.reshape(b, l // patch_size, patch_size) == 0).all(dim=-1).to(device)  # (B, num_patches)
    return one(src_padded_ids), one(dst_padded_ids)


#        Usage inside compute_src_dst_node_temporal_embeddings:
#            src_pad, dst_pad = build_patch_pad_mask(src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids, self.patch_size, self.device)
#            pad_mask = torch.cat([src_pad, dst_pad], dim=1)
#            for transformer in self.transformers:
#                patches_data = transformer(patches_data, key_padding_mask=pad_mask)
#
#    (c) Replace the two torch.mean(...) pooling lines by a masked mean:
def masked_mean(x: torch.Tensor, pad: torch.Tensor):
    valid = (~pad).unsqueeze(-1).float()  # (B, P, 1)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
#            src_patches_data = masked_mean(patches_data[:, :src_num_patches, :], src_pad)
#            dst_patches_data = masked_mean(patches_data[:, src_num_patches:src_num_patches + dst_num_patches, :], dst_pad)


# ----------------------------------------------------------------------------------------------
# 3. TRAINING: causal historical negatives + K negatives + softmax loss (closer to MRR)
#    Put the class in utils/utils.py; use the loss function in train_link_prediction.py.
# ----------------------------------------------------------------------------------------------
class HistoricalNegativeSampler:
    """For a training edge (s, d, t): negatives are destinations s interacted with strictly before t
    (excluding d), mixed with uniformly random destinations. Fully causal."""

    def __init__(self, src, dst, times, seed: int = 0):
        order = np.lexsort((times, src))  # by src, then time
        s, d, t = src[order], dst[order], times[order]
        cuts = np.flatnonzero(np.diff(s)) + 1
        self.by_src = {}
        for s_grp, d_grp, t_grp in zip(np.split(s, cuts), np.split(d, cuts), np.split(t, cuts)):
            self.by_src[int(s_grp[0])] = (t_grp, d_grp)
        self.unique_dst = np.unique(dst)
        self.rng = np.random.RandomState(seed)

    def sample(self, batch_src, batch_dst, batch_times, k: int, hist_frac: float = 0.5):
        """returns (B, k) int64 array of negative destination ids"""
        out = self.rng.choice(self.unique_dst, size=(len(batch_src), k))  # random part / fallback
        n_hist = int(round(k * hist_frac))
        if n_hist == 0:
            return out.astype(np.int64)
        for i, (s, d, t) in enumerate(zip(batch_src, batch_dst, batch_times)):
            grp = self.by_src.get(int(s))
            if grp is None:
                continue
            times, dsts = grp
            cand = dsts[: np.searchsorted(times, t, side='left')]
            cand = cand[cand != d]
            if len(cand) > 0:
                out[i, :n_hist] = self.rng.choice(cand, size=n_hist, replace=len(cand) < n_hist)
        return out.astype(np.int64)


def multi_negative_loss(model, batch_src, batch_dst, batch_times, batch_neg_dst, pair_history=None):
    """model = nn.Sequential(dynamic_backbone, link_predictor). batch_neg_dst: (B, K) ndarray.
    If pair_history is given, model[1] must be a RecurrenceMergeLayer (section 4).
    Cost grows ~ (1 + K)x per step, because DyGFormer's co-occurrence features depend on the (src, dst) pair."""
    def logit(dst_ids):
        s_emb, d_emb = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src, dst_node_ids=dst_ids, node_interact_times=batch_times)
        if pair_history is None:
            return model[1](input_1=s_emb, input_2=d_emb).squeeze(-1)  # raw logit, no sigmoid
        extra = torch.from_numpy(pair_history.features(batch_src, dst_ids, batch_times)).to(s_emb.device)
        return model[1](input_1=s_emb, input_2=d_emb, extra=extra).squeeze(-1)

    logits = torch.stack([logit(batch_dst)] + [logit(batch_neg_dst[:, j]) for j in range(batch_neg_dst.shape[1])], dim=1)  # (B, 1+K)
    return F.cross_entropy(logits, torch.zeros(len(batch_src), dtype=torch.long, device=logits.device))


#    In train_link_prediction.py, replace the negative sampling / BCE block by:
#        hist_sampler = HistoricalNegativeSampler(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)   # once, before the run loop
#        ...
#        batch_neg_dst = hist_sampler.sample(batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, k=args_k, hist_frac=0.5)
#        loss = multi_negative_loss(model, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, torch.from_numpy(batch_neg_dst))
#    and drop the per-batch sklearn AP/AUC computation (it is slow and not needed for model selection, which uses validation MRR).


# ----------------------------------------------------------------------------------------------
# 4. RECURRENCE FEATURES for the scoring head (do this AFTER task 3, otherwise "has this pair
#    occurred before" becomes a shortcut against purely random training negatives).
#    Causal: only edges strictly before the query time are used, same as the neighbour sampler.
# ----------------------------------------------------------------------------------------------
class PairHistory:
    def __init__(self, src, dst, times):  # pass full_data arrays (chronological)
        self.h = {}
        for s, d, t in zip(src, dst, times):
            self.h.setdefault((int(s), int(d)), []).append(t)
        self.h = {k: np.asarray(v) for k, v in self.h.items()}

    def features(self, srcs, dsts, ts):
        """(B, 3): [seen_before, log1p(count_before), log1p(time_since_last)]"""
        out = np.zeros((len(srcs), 3), dtype=np.float32)
        for i, (s, d, t) in enumerate(zip(srcs, dsts, ts)):
            arr = self.h.get((int(s), int(d)))
            if arr is None:
                continue
            k = np.searchsorted(arr, t, side='left')
            if k > 0:
                out[i] = (1.0, np.log1p(k), np.log1p(t - arr[k - 1]))
        return out


class RecurrenceMergeLayer(nn.Module):
    """Replaces MergeLayer(...): scores a pair from both embeddings plus extra scalar features."""

    def __init__(self, emb_dim: int, extra_dim: int = 3, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.fc1 = nn.Linear(2 * emb_dim + extra_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_1, input_2, extra):
        return self.fc2(F.relu(self.fc1(torch.cat([input_1, input_2, extra], dim=-1))))

# Hooking this in: build `pair_history = PairHistory(full_data.src_node_ids, full_data.dst_node_ids, full_data.node_interact_times)` once,
# create `link_predictor = RecurrenceMergeLayer(emb_dim=args.output_dim)`, pass pair_history to multi_negative_loss in training,
# and use evaluate_dygformer_link_prediction (section 5) instead of evaluate_model_link_prediction.


# ----------------------------------------------------------------------------------------------
# 5. TGB EVALUATION HOOK + DIAGNOSTICS  (written against your evaluate_models_utils.py, DyGFormer branch only)
#    Differences from the original function:
#      - optional recurrence features for positives AND for every (src, negative dst) pair
#      - use_logits=True ranks on raw logits instead of sigmoid probabilities (see note below)
#      - return_details=True also returns per-query (mrr, positive_seen_before, n_ties) for diagnostics
#    NOTE on ties: the original ranks sigmoid() outputs in float32. A saturated sigmoid gives exactly 1.0 (or 0.0) for many
#    candidates; TGB counts ties as half a rank, which hurts MRR. Ranking on logits keeps the same order but breaks those ties.
#    Check n_ties in the details first: if it is ~0 everywhere, this changes nothing.
# ----------------------------------------------------------------------------------------------
def evaluate_dygformer_link_prediction(model, neighbor_sampler, evaluate_idx_data_loader, evaluate_neg_edge_sampler, evaluate_data,
                                       eval_stage, eval_metric_name, evaluator, pair_history=None, use_logits=False, return_details=False, diag_history=None):
    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()
    metrics, details = [], []
    with torch.no_grad():
        for idx in evaluate_idx_data_loader:
            idx = idx.numpy()
            src, dst, ts = evaluate_data.src_node_ids[idx], evaluate_data.dst_node_ids[idx], evaluate_data.node_interact_times[idx]
            # same id mapping as the original: TGB ids are 1 lower than DyGLib ids
            neg_dst = np.array(evaluate_neg_edge_sampler.query_batch(pos_src=src - 1, pos_dst=dst - 1, pos_timestamp=ts, split_mode=eval_stage)) + 1
            n = neg_dst.shape[1]
            rep_src, rep_ts, flat_neg = np.repeat(src, n, axis=0), np.repeat(ts, n, axis=0), neg_dst.flatten()

            s_e, d_e = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src, dst_node_ids=dst, node_interact_times=ts)
            ns_e, nd_e = model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=rep_src, dst_node_ids=flat_neg, node_interact_times=rep_ts)

            if pair_history is None:
                pos = model[1](input_1=s_e, input_2=d_e)
                neg = model[1](input_1=ns_e, input_2=nd_e)
                # baseline model (no recurrence features): optionally pass diag_history=PairHistory(...) just to get the recurring/new split
                pos_seen = diag_history.features(src, dst, ts)[:, 0] if (return_details and diag_history is not None) else np.zeros(len(src))
            else:
                pos_feat = pair_history.features(src, dst, ts)
                neg_feat = pair_history.features(rep_src, flat_neg, rep_ts)
                pos = model[1](input_1=s_e, input_2=d_e, extra=torch.from_numpy(pos_feat).to(s_e.device))
                neg = model[1](input_1=ns_e, input_2=nd_e, extra=torch.from_numpy(neg_feat).to(ns_e.device))
                pos_seen = pos_feat[:, 0]
            pos, neg = pos.squeeze(-1), neg.squeeze(-1)
            if not use_logits:
                pos, neg = pos.sigmoid(), neg.sigmoid()
            pos, neg = pos.cpu().numpy(), neg.cpu().numpy()

            for i in range(len(src)):
                p, q = pos[i: i + 1], neg[i * n: (i + 1) * n]
                mrr = evaluator.eval({"y_pred_pos": p, "y_pred_neg": q, "eval_metric": [eval_metric_name]})[eval_metric_name]
                metrics.append({eval_metric_name: mrr})
                if return_details:
                    details.append((mrr, float(pos_seen[i]), int((q >= p).sum() - (q > p).sum())))
    return (metrics, details) if return_details else metrics


def summarize_details(details):
    """details: list of (mrr, positive_seen_before, n_ties). Prints MRR split by recurring vs. new positive edge and tie statistics."""
    d = np.asarray(details, dtype=float)
    for name, mask in [("all", np.ones(len(d), bool)), ("recurring positive edge", d[:, 1] == 1), ("new positive edge", d[:, 1] == 0)]:
        if mask.any():
            print(f"{name:26s} n={mask.sum():8d}  MRR={d[mask, 0].mean():.4f}")
    print(f"queries with ties: {(d[:, 2] > 0).mean():.2%}, mean ties: {d[:, 2].mean():.3f}")
    # NOTE: for the baseline model pass diag_history=PairHistory(...) to evaluate_dygformer_link_prediction, otherwise the split is all zeros.
