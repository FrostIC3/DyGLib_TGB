"""
Run from the DyGLib_TGB repo root (needs torch):   python test_dygformer_patches.py

Test 1 (co-occurrence): count_nodes_appearances must equal a plain-numpy reference of the ORIGINAL loop semantics.
        -> should PASS both before and after --fast-cooc (after = the vectorised version is faithful).
        NOTE: with --log-counts this test is expected to FAIL by design (it compares raw counts).
Test 2 (padding invariance): the embedding of pair A must not change when A is batched with a pair whose history is longer.
        -> expected to FAIL on the unpatched model (padding leaks into attention + mean pooling), PASS after --mask.
"""
import numpy as np
import torch

from models.DyGFormer import DyGFormer, NeighborCooccurrenceEncoder
from utils.utils import NeighborSampler


def reference_counts(s_ids, d_ids):
    out_s, out_d = [], []
    for s, d in zip(s_ids, d_ids):
        su, si, sc = np.unique(s, return_inverse=True, return_counts=True)
        du, di, dc = np.unique(d, return_inverse=True, return_counts=True)
        sm, dm = dict(zip(su, sc)), dict(zip(du, dc))
        out_s.append(np.stack([sc[si], [dm.get(x, 0) for x in s]], 1))
        out_d.append(np.stack([[sm.get(x, 0) for x in d], dc[di]], 1))
    out_s, out_d = np.stack(out_s).astype(np.float32), np.stack(out_d).astype(np.float32)
    out_s[s_ids == 0] = 0
    out_d[d_ids == 0] = 0
    return out_s, out_d


def test_cooc():
    enc = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=8, device='cpu')
    rng = np.random.default_rng(0)
    for _ in range(30):
        B, Ls, Ld = rng.integers(1, 6), rng.integers(2, 20), rng.integers(2, 20)
        s = rng.integers(0, 8, (B, Ls)).astype(np.int64)
        d = rng.integers(0, 8, (B, Ld)).astype(np.int64)
        s[:, 0] = rng.integers(1, 8, B)
        d[:, 0] = rng.integers(1, 8, B)
        got_s, got_d = enc.count_nodes_appearances(s, d)
        ref_s, ref_d = reference_counts(s, d)
        assert np.allclose(got_s.cpu().numpy(), ref_s) and np.allclose(got_d.cpu().numpy(), ref_d)
    print("Test 1 PASS: co-occurrence counts match the reference")


def build_model():
    torch.manual_seed(0)
    n_nodes, n_edges, t_query = 12, 40, 100.0
    rng = np.random.default_rng(0)
    adj = [[] for _ in range(n_nodes + 1)]      # index 0 stays empty
    eid = 1
    def add(u, v, t):
        nonlocal eid
        adj[u].append((v, eid, t)); adj[v].append((u, eid, t)); eid += 1
    add(1, 2, 1.0); add(1, 3, 2.0)                                   # short history for nodes 1, 2, 3
    for k in range(14):                                              # long history for node 4
        add(4, 5 + (k % 6), 3.0 + k)
    node_feats = rng.normal(size=(n_nodes + 1, 8)).astype(np.float32); node_feats[0] = 0
    edge_feats = rng.normal(size=(eid, 4)).astype(np.float32); edge_feats[0] = 0
    sampler = NeighborSampler(adj_list=adj, sample_neighbor_strategy='recent', seed=None)
    model = DyGFormer(node_raw_features=node_feats, edge_raw_features=edge_feats, neighbor_sampler=sampler,
                      time_feat_dim=8, channel_embedding_dim=8, output_dim=8, patch_size=1, num_layers=2, num_heads=2,
                      dropout=0.0, max_input_sequence_length=16, device='cpu')
    model.eval()
    return model, t_query


def test_padding_invariance():
    model, t = build_model()
    with torch.no_grad():
        a_s, a_d = model.compute_src_dst_node_temporal_embeddings(np.array([1]), np.array([2]), np.array([t]))
        # pair A batched with pair B whose source (node 4) has a much longer history -> A gets padded further
        b_s, b_d = model.compute_src_dst_node_temporal_embeddings(np.array([1, 4]), np.array([2, 5]), np.array([t, t]))
    diff = max((a_s[0] - b_s[0]).abs().max().item(), (a_d[0] - b_d[0]).abs().max().item())
    print(f"max |embedding(alone) - embedding(in batch)| = {diff:.3e}")
    if diff < 1e-5:
        print("Test 2 PASS: embeddings do not depend on batch padding")
    else:
        print("Test 2 FAIL: embeddings depend on batch padding (expected on the UNPATCHED model; apply --mask)")


if __name__ == '__main__':
    test_cooc()
    test_padding_invariance()
