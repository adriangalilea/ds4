#!/usr/bin/env python3
"""Run released DSpark Python blocks on CPU with the measured GGUF weight recipe.

TileLang operators have explicit PyTorch equivalents. This is a numerical
semantic oracle, not a bit-exact replica of Metal reduction trees. Experts
are materialized on demand so the three blocks fit a 24-GiB development Mac.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from deepseek41_quantize import SourceDB, NativeQuantizer, build_plan, validate_scales


def act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=None, inplace=False):
    assert inplace and scale_fmt == "ue8m0"
    blocks = x.float().reshape(*x.shape[:-1], -1, block_size)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    value = ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale)
    x.copy_(value.reshape(x.shape).to(x.dtype))
    return x


def hc_split(mixes, scale, base, hc=4, iters=20, eps=1e-6):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2*hc] * scale[1] + base[hc:2*hc])
    comb = (mixes[..., 2*hc:] * scale[2] + base[2*hc:]).reshape(*mixes.shape[:-1], hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, sink, ids, scale):
    assert q.shape[0] == 1
    values = kv[:, ids[0].long()]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), values.float()) * scale
    maximum = scores.amax(-1, keepdim=True)
    weights = (scores - maximum).exp()
    denominator = weights.sum(-1, keepdim=True) + (sink[None, None, :, None] - maximum).exp()
    return (torch.einsum("bshk,bskd->bshd", weights, values.float()) / denominator).to(q.dtype)


def unsupported(*args, **kwargs):
    raise AssertionError("quantized CUDA GEMM must be replaced by the GGUF-weight CPU linear")


def load_reference(path):
    kernel = types.ModuleType("kernel")
    kernel.act_quant = act_quant
    kernel.hc_split_sinkhorn = hc_split
    kernel.sparse_attn = sparse_attn
    kernel.fp4_act_quant = kernel.fp4_gemm = kernel.fp8_gemm = unsupported
    sys.modules["kernel"] = kernel
    sys.path.insert(0, str(path / "inference"))
    spec = importlib.util.spec_from_file_location("deepseek41_reference", path / "inference/model.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    ref.default_dtype = torch.bfloat16
    ref.linear = lambda x, w, bias=None: F.linear(x.float(), w.float(), bias).to(x.dtype)
    # The released grouped output projection reads weight directly in einsum.
    # Preserve its GGUF float values while rounding the output to the input dtype.
    original_einsum = torch.einsum
    def einsum(equation, *args):
        if len(args) == 2 and args[0].dtype != args[1].dtype:
            return original_einsum(equation, *(x.float() for x in args)).to(args[0].dtype)
        return original_einsum(equation, *args)
    torch.einsum = einsum
    return ref


class Weights:
    def __init__(self, source):
        self.db = SourceDB(str(source), index_validator=lambda _: None,
            scale_validator=lambda t: validate_scales(t, True),
            tensor_filter=lambda n: n.startswith("mtp.") or n == "head.weight")
        self.q = NativeQuantizer(str(ROOT / "gguf-tools/libds4quants.dylib"))
        plan = build_plan(self.db, json.loads((source / "config.json").read_text()), "q4", True)
        self.types = {}
        self.types["head.weight"] = 8
        for item in plan:
            if item.is_expert:
                for expert in range(item.expert_count):
                    self.types[item.source.format(expert=expert)] = item.qtype
            else:
                self.types[item.source] = item.qtype

    def get(self, name, row_start=0, row_count=None):
        source = self.q.to_f32(self.db, name, row_start, row_count)
        qt = self.types[name]
        encoded = self.q.encode(source, qt)
        if name == "head.weight":
            if row_start == 0:
                self.head_hash = hashlib.sha256()
            self.head_hash.update(encoded)
        if qt == 0:
            values = np.frombuffer(encoded, dtype="<f4").copy()
        elif qt == 1:
            values = np.frombuffer(encoded, dtype="<f2").astype(np.float32)
        else:
            values = dequantize(np.frombuffer(encoded, np.uint8), GGMLQuantizationType(qt))
        return torch.from_numpy(values.reshape(source.shape).copy())


class SharedHead(torch.nn.Module):
    def __init__(self, weights, vocab):
        super().__init__()
        self.weights, self.vocab = weights, vocab

    def forward(self, x, full_logits=False):
        if not full_logits:
            x = x[:, -1]
        output = torch.empty(*x.shape[:-1], self.vocab, dtype=torch.float32)
        for start in range(0, self.vocab, 2048):
            count = min(2048, self.vocab - start)
            output[..., start:start+count] = F.linear(x.float(), self.weights.get("head.weight", start, count))
        return output


def make_blocks(ref, args, weights):
    blocks = []
    for stage in range(3):
        with torch.device("meta"):
            block = ref.DSparkBlock(args.n_layers + stage, args)
        for name, param in list(block.named_parameters()):
            if ".experts." in name:
                continue
            parent, leaf = name.rsplit(".", 1) if "." in name else ("", name)
            setattr(block.get_submodule(parent), leaf,
                torch.nn.Parameter(weights.get(f"mtp.{stage}.{name}"), requires_grad=False))
        for name, module in block.named_modules():
            if isinstance(module, ref.Linear) and ".experts." in name:
                source = f"mtp.{stage}.{name}.weight"
                def forward(x, source=source):
                    return F.linear(x.float(), weights.get(source)).to(x.dtype)
                module.forward = forward
        attn = block.attn
        attn.window_kv_cache = torch.zeros(1, 128, args.head_dim, dtype=torch.bfloat16)
        ref.precompute_freqs_cis.cache_clear()
        attn.freqs_cis = ref.precompute_freqs_cis(args.rope_head_dim, args.max_seq_len,
            0, args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        blocks.append(block)
    return blocks


def read(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(shape))


def compare(folder, label, value, reports):
    expected = read(folder / (label + ".f32"), value.shape).float()
    value = value.float()
    delta = value - expected
    report = dict(name=label, max_abs=delta.abs().max().item(),
        rms=delta.square().mean().sqrt().item(),
        relative_rms=(delta.square().mean() / expected.square().mean().clamp_min(1e-30)).sqrt().item(),
        cosine=F.cosine_similarity(value.flatten(), expected.flatten(), dim=0).item())
    reports.append(report)
    print(json.dumps(report), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-sha256", help="expected hash of the target GGUF Q8 head payload")
    opts = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    ref = load_reference(opts.reference)
    config = json.loads((opts.reference / "inference/config.json").read_text())
    config.update(max_batch_size=1, max_seq_len=4096, dtype="bf16", expert_dtype=None, temperature=0)
    args = ref.ModelArgs(**config)
    weights = Weights(opts.hf)
    blocks = make_blocks(ref, args, weights)
    blocks[-1].head = SharedHead(weights, args.vocab_size)
    all_reports = {}
    for folder in sorted(opts.dump.iterdir(), key=lambda p: int(p.name)):
        meta = json.loads((folder / "meta.json").read_text())
        pos = meta["position"]
        hidden = read(folder / "hidden.f32", (1, 128, 3 * args.dim)).to(torch.bfloat16)
        x = read(folder / "0-embedding.f32", (1, 5, 4, args.dim)).to(torch.bfloat16)
        main_x = blocks[0].main_norm(blocks[0].main_proj(hidden))
        # Match the scalar current-token projection used after seeding.
        current = blocks[0].main_norm(blocks[0].main_proj(hidden[:, -1:]))
        reports = []
        compare(folder, "0-main", current, reports)
        pre = ref.make_identity_pre_mix(x, 4)
        for stage, block in enumerate(blocks):
            attn = block.attn
            kv = attn.kv_norm(attn.wkv(main_x))
            ref.apply_rotary_emb(kv[..., -args.rope_head_dim:], attn.freqs_cis[pos-127:pos+1])
            act_quant(kv, 32, "ue8m0", inplace=True)
            attn.window_kv_cache[:, torch.arange(pos-127, pos+1) % 128] = kv
            hooks = [
                block.attn_norm.register_forward_hook(lambda m, a, out, s=stage: compare(folder, f"{s}-attn_input", out, reports)),
                block.attn.register_forward_hook(lambda m, a, out, s=stage: compare(folder, f"{s}-attn_output", out, reports)),
                block.ffn_norm.register_forward_hook(lambda m, a, out, s=stage: compare(folder, f"{s}-ffn_input", out, reports)),
            ]
            x, pre = block(x, pos, pre, current)
            for hook in hooks:
                hook.remove()
            compare(folder, f"{stage}-residual", x, reports)
        collapsed = blocks[-1].hc_pre(x, pre)
        compare(folder, "2-collapsed", collapsed, reports)
        base = read(folder / "base_logits.f32", (1, 5, args.vocab_size))
        reference_base = blocks[-1].head(blocks[-1].norm(collapsed), full_logits=True)
        head_hash = weights.head_hash.hexdigest()
        if opts.head_sha256:
            assert head_hash == opts.head_sha256, (head_hash, opts.head_sha256)
        print(json.dumps(dict(target_head_sha256=head_hash)), flush=True)
        compare(folder, "base_logits", reference_base, reports)
        previous = torch.tensor([meta["seed"]])
        proposed, embeds = [], []
        for i in range(5):
            bias, embed = blocks[-1].markov_head(previous)
            previous = (base[:, i] + bias).argmax(-1)
            proposed.append(previous.item())
            embeds.append(embed)
        confidence = blocks[-1].confidence_head(collapsed, torch.stack(embeds, dim=1))
        compare(folder, "confidence", confidence, reports)
        print(json.dumps(dict(position=pos, metal_proposals=meta["proposals"],
            reference_markov_on_metal_base=proposed)), flush=True)
        previous = torch.tensor([meta["seed"]])
        complete = []
        for i in range(5):
            bias, _ = blocks[-1].markov_head(previous)
            previous = (reference_base[:, i] + bias).argmax(-1)
            complete.append(previous.item())
        print(json.dumps(dict(position=pos, reference_proposals=complete,
            exact_proposals=complete == meta["proposals"])), flush=True)
        all_reports[folder.name] = dict(intermediates=reports, reference_proposals=complete,
            metal_proposals=meta["proposals"], exact_proposals=complete == meta["proposals"],
            target_head_sha256=head_hash)
    opts.output.write_text(json.dumps(all_reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
