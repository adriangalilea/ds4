"""Measure native/quantized reference proposals on every captured target position.

Uses the same released blocks and CPU operator adapter as the three-position
oracle. Weight caching changes only repeated I/O and quantization work.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from reference_deepseek41_dspark import Weights, act_quant, load_reference, make_blocks, read


class CachedWeights(Weights):
    def __init__(self, source, native, cache_gib):
        super().__init__(source, native)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.limit = int(cache_gib * 1024**3)

    def get(self, name, row_start=0, row_count=None):
        key = (name, row_start, row_count)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        value = super().get(name, row_start, row_count)
        size = value.numel() * value.element_size()
        if size <= self.limit and name != "head.weight":
            while self.cache and self.cache_bytes + size > self.limit:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= old.numel() * old.element_size()
            self.cache[key] = value
            self.cache_bytes += size
        return value


class CachedHead(torch.nn.Module):
    def __init__(self, weights, vocab, dim, expected_hash):
        super().__init__()
        self.weight = torch.empty(vocab, dim, dtype=torch.float32)
        for start in range(0, vocab, 2048):
            count = min(2048, vocab - start)
            self.weight[start:start + count] = weights.get("head.weight", start, count)
        self.sha256 = weights.head_hash.hexdigest()
        assert self.sha256 == expected_hash, self.sha256

    def forward(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        return F.linear(x.float(), self.weight)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--hf", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--metal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--head-sha256", required=True)
    p.add_argument("--native-drafter", action="store_true")
    p.add_argument("--cache-gib", type=float, default=8)
    p.add_argument("--threads", type=int, default=8)
    opts = p.parse_args()
    torch.set_num_threads(opts.threads)
    torch.set_default_dtype(torch.bfloat16)
    ref = load_reference(opts.reference)
    config = json.loads((opts.reference / "inference/config.json").read_text())
    config.update(max_batch_size=1, max_seq_len=4096, dtype="bf16", expert_dtype=None, temperature=0)
    args = ref.ModelArgs(**config)
    weights = CachedWeights(opts.hf, opts.native_drafter, opts.cache_gib)
    blocks = make_blocks(ref, args, weights)
    blocks[-1].head = CachedHead(weights, args.vocab_size, args.dim, opts.head_sha256)
    meta = json.loads((opts.dump / "meta.json").read_text())
    assert meta["rows"] == 383 and meta["first_generation_row"] == 127
    generated = meta["generated"]
    assert len(generated) == 256
    metal = [json.loads(line) for line in opts.metal.read_text().splitlines() if line.startswith("{")]
    assert [row["seed"] for row in metal] == generated
    hidden = read(opts.dump / "mean.f32", (1, 383, 3 * args.dim)).to(torch.bfloat16)
    positions = torch.arange(meta["start"], meta["start"] + 383)
    # Initial window uses the same 128-row projection as the three-position oracle.
    initial = blocks[0].main_norm(blocks[0].main_proj(hidden[:, :128]))
    for block in blocks:
        kv = block.attn.kv_norm(block.attn.wkv(initial))
        ref.apply_rotary_emb(kv[..., -args.rope_head_dim:], block.attn.freqs_cis[positions[:128]])
        act_quant(kv, 32, "ue8m0", inplace=True)
        block.attn.window_kv_cache[:, positions[:128] % 128] = kv
    rows = []
    first = 0
    hist = [0] * 6
    started = time.monotonic()
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    with opts.output.open("x") as out:
        for i in range(255):
            pos = meta["start"] + 127 + i
            seed = generated[i]
            embedding = torch.stack([weights.get("embed.weight", token, 1)[0]
                                     for token in [seed] + [args.dspark_noise_token_id] * 4])
            x = embedding.to(torch.bfloat16)[None, :, None, :].repeat(1, 1, 4, 1)
            current = blocks[0].main_norm(blocks[0].main_proj(hidden[:, 127 + i:128 + i]))
            pre = ref.make_identity_pre_mix(x, 4)
            for block in blocks:
                x, pre = block(x, pos, pre, current)
            collapsed = blocks[-1].hc_pre(x, pre)
            base = blocks[-1].head(blocks[-1].norm(collapsed), full_logits=True)
            previous = torch.tensor([seed])
            proposals, embeds = [], []
            for j in range(5):
                bias, embed = blocks[-1].markov_head(previous)
                previous = (base[:, j] + bias).argmax(-1)
                proposals.append(previous.item())
                embeds.append(embed)
            confidence = blocks[-1].confidence_head(collapsed, torch.stack(embeds, dim=1))[0].tolist()
            first += int(proposals[0] == generated[i + 1])
            if i < 251:
                n = 0
                while n < 5 and proposals[n] == generated[i + n + 1]:
                    n += 1
                hist[n] += 1
            row = dict(i=i, position=pos, seed=seed, proposals=proposals, confidence=confidence,
                       metal_proposals=metal[i]["proposals"], first_matches_so_far=first,
                       elapsed_sec=time.monotonic() - started)
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
    summary = dict(native_drafter=opts.native_drafter, first_matches=first, first_opportunities=255,
                   first_agreement=first / 255, prefix_histogram=hist,
                   mean_prefix=sum(i * n for i, n in enumerate(hist)) / 251,
                   metal_first_matches=sum(r["proposals"][0] == generated[i + 1] for i, r in enumerate(metal[:255])),
                   first_proposals_changed=sum(r["proposals"][0] != r["metal_proposals"][0] for r in rows),
                   elapsed_sec=time.monotonic() - started, head_sha256=blocks[-1].head.sha256)
    opts.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
