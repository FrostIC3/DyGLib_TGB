"""
Apply independent patches to DyGLib_TGB/models/DyGFormer.py.

    python apply_dygformer_patches.py --path models/DyGFormer.py --fast-cooc
    python apply_dygformer_patches.py --path models/DyGFormer.py --mask
    python apply_dygformer_patches.py --path models/DyGFormer.py --log-counts   (needs --fast-cooc, applied together or after)

Flags:
  --fast-cooc   vectorised co-occurrence counts (same numbers as the original loop, much faster, chunked to bound memory)
  --mask        key_padding_mask in attention + masked mean pooling (removes dependence of embeddings on batch padding)
  --log-counts  log1p on co-occurrence counts before the MLP (changes the model; treat as an experiment)

Safety: every anchor must match exactly once, otherwise nothing is written. A backup <file>.orig is created on first use.
Use one git branch per flag so each change can be evaluated on its own.
"""
import argparse
import os
import shutil
import sys

FAST_COOC = '''    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        [patched: fast-cooc] vectorised version of the original per-sample loop.
        Returns two Tensors, shape (batch_size, src_max_seq_length, 2) and (batch_size, dst_max_seq_length, 2).
        """
        s_all = torch.from_numpy(src_padded_nodes_neighbor_ids).to(self.device)
        d_all = torch.from_numpy(dst_padded_nodes_neighbor_ids).to(self.device)
        max_len = max(s_all.shape[1], d_all.shape[1])
        # bound the (chunk, L, L) comparison tensors to ~5e7 elements
        chunk = max(1, int(5e7 // (max_len * max_len)))
        src_out, dst_out = [], []
        for start in range(0, s_all.shape[0], chunk):
            s = s_all[start: start + chunk]
            d = d_all[start: start + chunk]
            s_in_s = (s.unsqueeze(2) == s.unsqueeze(1)).sum(-1)
            d_in_d = (d.unsqueeze(2) == d.unsqueeze(1)).sum(-1)
            s_in_d = (s.unsqueeze(2) == d.unsqueeze(1)).sum(-1)
            d_in_s = (d.unsqueeze(2) == s.unsqueeze(1)).sum(-1)
            src_app = torch.stack([s_in_s, s_in_d], dim=-1).float()
            dst_app = torch.stack([d_in_s, d_in_d], dim=-1).float()
            # padded positions (id 0) get zero appearances
            src_app[s == 0] = 0.0
            dst_app[d == 0] = 0.0
            src_out.append(src_app)
            dst_out.append(dst_app)
        src_app, dst_app = torch.cat(src_out, dim=0), torch.cat(dst_out, dim=0)
        if __LOG_COUNTS__:
            src_app, dst_app = torch.log1p(src_app), torch.log1p(dst_app)
        return src_app, dst_app

'''

MASK_LOOP_OLD = '''        for transformer in self.transformers:
            patches_data = transformer(patches_data)
'''
MASK_LOOP_NEW = '''        # [patched: mask] a patch is padding only if all its node ids are 0; the first patch holds the target node, so it is never padding
        src_pad_mask = torch.from_numpy(src_padded_nodes_neighbor_ids.reshape(batch_size, src_num_patches, self.patch_size) == 0).all(dim=-1).to(self.device)
        dst_pad_mask = torch.from_numpy(dst_padded_nodes_neighbor_ids.reshape(batch_size, dst_num_patches, self.patch_size) == 0).all(dim=-1).to(self.device)
        pad_mask = torch.cat([src_pad_mask, dst_pad_mask], dim=1)
        for transformer in self.transformers:
            patches_data = transformer(patches_data, key_padding_mask=pad_mask)
'''
MASK_SRC_MEAN_OLD = "        src_patches_data = torch.mean(src_patches_data, dim=1)\n"
MASK_SRC_MEAN_NEW = '''        src_valid = (~src_pad_mask).unsqueeze(-1).float()
        src_patches_data = (src_patches_data * src_valid).sum(dim=1) / src_valid.sum(dim=1).clamp(min=1.0)
'''
MASK_DST_MEAN_OLD = "        dst_patches_data = torch.mean(dst_patches_data, dim=1)\n"
MASK_DST_MEAN_NEW = '''        dst_valid = (~dst_pad_mask).unsqueeze(-1).float()
        dst_patches_data = (dst_patches_data * dst_valid).sum(dim=1) / dst_valid.sum(dim=1).clamp(min=1.0)
'''
MASK_SIG_OLD = "    def forward(self, inputs: torch.Tensor):\n"
MASK_SIG_NEW = "    def forward(self, inputs: torch.Tensor, key_padding_mask: torch.Tensor = None):\n"
MASK_ATT_OLD = "hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)"
MASK_ATT_NEW = "hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs, key_padding_mask=key_padding_mask, need_weights=False)[0].transpose(0, 1)"


def replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT: anchor for '{what}' found {n} times (expected exactly 1). Nothing was written.")
    return text.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', default='models/DyGFormer.py')
    ap.add_argument('--fast-cooc', action='store_true')
    ap.add_argument('--mask', action='store_true')
    ap.add_argument('--log-counts', action='store_true')
    a = ap.parse_args()
    if a.log_counts and not a.fast_cooc:
        sys.exit("--log-counts is implemented inside the fast co-occurrence method; pass --fast-cooc too.")
    if not (a.fast_cooc or a.mask):
        sys.exit("nothing to do: pass --fast-cooc and/or --mask")

    text = open(a.path).read()
    if '[patched:' in text:
        sys.exit("ABORT: file already contains patches. Restore from the .orig backup or git first.")

    if a.fast_cooc:
        start = text.find("    def count_nodes_appearances(")
        end = text.find("    def forward(self, src_padded_nodes_neighbor_ids", start)
        if start < 0 or end < 0 or text.count("    def count_nodes_appearances(") != 1:
            sys.exit("ABORT: could not locate count_nodes_appearances. Nothing was written.")
        method = FAST_COOC.replace("__LOG_COUNTS__", "True" if a.log_counts else "False")
        text = text[:start] + method + text[end:]

    if a.mask:
        text = replace_once(text, MASK_LOOP_OLD, MASK_LOOP_NEW, "transformer loop")
        text = replace_once(text, MASK_SRC_MEAN_OLD, MASK_SRC_MEAN_NEW, "src mean pooling")
        text = replace_once(text, MASK_DST_MEAN_OLD, MASK_DST_MEAN_NEW, "dst mean pooling")
        text = replace_once(text, MASK_SIG_OLD, MASK_SIG_NEW, "TransformerEncoder.forward signature")
        text = replace_once(text, MASK_ATT_OLD, MASK_ATT_NEW, "attention call")

    compile(text, a.path, 'exec')  # syntax check before writing anything
    backup = a.path + '.orig'
    if not os.path.exists(backup):
        shutil.copy(a.path, backup)
    open(a.path, 'w').write(text)
    print(f"patched {a.path} (fast_cooc={a.fast_cooc}, mask={a.mask}, log_counts={a.log_counts}); backup at {backup}")


if __name__ == '__main__':
    main()
