# TGB leaderboard target — beat TPNet on tgbl-wiki-v2

## Header

```
TARGET_DATASET      = tgbl-wiki  (== "tgbl-wiki-v2" on the leaderboard; 157,474 edges, 9,227 nodes)
BASELINE_VAL_MRR    = <pending T0.2>
BASELINE_TEST_MRR   = <pending T0.2>
LEADERBOARD_TOP     = TPNet  test 0.827 +/- 0.001  |  val 0.842 +/- 0.001
COMPUTE_BUDGET      = remote GPU server (this laptop is CPU-only: 8 cores, ~3 GB free RAM)
```

Gap to close, from the official board (package >= 0.7.5), read 2026-09-30:

| Rank | Method | Test MRR | Val MRR | Date |
|---|---|---|---|---|
| 1 | **TPNet** (official) | 0.827 ± 0.001 | 0.842 ± 0.001 | Aug 14 2025 |
| 2 | Heuristic(LocalGlobal) | 0.821 | 0.842 | Jan 20 2026 |
| 3 | HyperEvent | 0.810 ± 0.002 | 0.824 ± 0.002 | Aug 14 2025 |
| 4 | **DyGFormer** (this repo) | **0.798 ± 0.004** | **0.816 ± 0.005** | Aug 22 2023 |
| 5 | NAT | 0.749 ± 0.010 | 0.773 ± 0.011 | Aug 22 2023 |
| 6 | Base3 | 0.743 | 0.727 | Aug 14 2025 |

- **Gap to close on test: 0.029** (0.798 → 0.827). On validation: 0.026.
- DyGFormer's own val→test drop is **0.018**, which is large. Worth diagnosing
  before adding capacity; a model that fits val but not test will not close the gap.
- Rank 2 is a **non-learnable heuristic** that ties TPNet on validation and is within
  0.006 on test. Non-learnable signal is strong on this dataset, which supports
  prioritising T2.1/T2.2/T2.3 (recurrence, popularity, score blending) over pure
  architecture work in T4.2.

## Correction to the prior advice

The advice circulated before this task reported "TPNet 0.842, HyperEvent 0.824,
DyGFormer 0.816 **test** MRR". Those are the **validation** column. Test MRR is
0.827 / 0.810 / 0.798. The quoted figures conflated the two columns.

## T0.6 — leaderboard rules

From https://tgb.complexdatalab.com/docs/leader_rules/ (read 2026-09-30):

- **Training split**: all data (edges, nodes, labels) usable for gradient descent,
  model tuning, model input.
- **Validation split**: for hyperparameter tuning only. **Gradient-based search on
  validation is not allowed.** Usable for the memory module.
- **Test split**: final evaluation only, no hyperparameter tuning. Usable for the
  memory module. No back-propagation from test information.
- **Submission**: 5 runs, report both validation and test. Google form linked from
  the TGB repo README. T5.3 stays a human step.

Answers to the three specific questions:

| Question | Answer |
|---|---|
| Seed ensembles allowed? | **Not addressed.** The rules page is silent. Do not assume; flag at T5.3. |
| Blending with non-learnable methods allowed? | **Not prohibited.** It is not gradient search on test, and memory modules may use val/test. Blending weights fitted on validation is ordinary hyperparameter tuning, which *is* allowed. Low risk, but confirm at T5.3. |
| Training on train+val allowed? | **Treat as BANNED.** Training on validation data is gradient descent on the validation split, which rule 2 forbids. It would also invalidate every comparison here. Do not do it. |

## Repo notes (verified, T0.1)

- Node ids are TGB ids **+ 1**; `0` is the padding node.
- `train_link_prediction.py` uses `nn.BCELoss` on one random negative per positive.
- `link_predictor` is `MergeLayer(input_dim1=output_dim, input_dim2=output_dim, hidden_dim=output_dim, output_dim=1)`;
  `model = nn.Sequential(backbone, link_predictor)`.
- `utils/load_configs.py:52` silently falls back to CPU when CUDA is absent, so
  `--gpu 0` on a machine without a GPU runs on CPU **without warning**. Check the
  log's `configuration is ...` line for `device=` before trusting any timing.
- **tgbl-wiki has no node features.** Verified: `node_raw` is shape `(9229, 1)` and
  all zeros, from the padding path at `utils/DataLoader.py:152`. So T4.2's
  "node identity embeddings" item is not optional here — the model currently has
  zero node-identity information beyond the sequence structure.
- Validation/test on tgbl-wiki use batch size 1 (all negatives), by design
  (`train_link_prediction.py:54-56`). The T0.4 padding diagnosis depends on this
  train-batch-size-200 vs eval-batch-size-1 mismatch.

## Patch tooling pre-flight (done, before first use)

- `apply_dygformer_patches.py --mask`: all 5 anchors match **exactly once** in
  `models/DyGFormer.py`, which defines its own `TransformerEncoder`. The script will
  not abort. It is atomic — the file is written only after every anchor resolves.
- `test_dygformer_patches.py` constructor signatures match the repo exactly:
  `NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim, device)`,
  `NeighborSampler(adj_list, sample_neighbor_strategy, time_scaling_factor, seed)`,
  `DyGFormer(...)` — all confirmed.
- Test 1's numpy reference is faithful: the original `count_nodes_appearances` does
  zero out padded positions before returning, so the reference matches it.
- Both files were originally sitting in `models/` (and `~/Downloads/`), not the repo
  root. Moved to the root in T0.1 as the task specifies.

## Log

- **T0.1** done: `--exp_name` added (default `base`) and threaded into all
  `logs/`, `saved_models/`, `saved_results/` paths in both
  `train_link_prediction.py` and `evaluate_link_prediction.py`; `.gitignore`
  patterns deepened by one level to match; patch files moved to repo root;
  `experiments/RESULTS.tsv` and this file created.
