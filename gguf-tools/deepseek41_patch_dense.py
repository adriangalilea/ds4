#!/usr/bin/env python3
"""Rewrite a V4.1 GGUF's dense tensors from a variant checkpoint that keeps the base's experts.

A variant such as an abliterated checkpoint edits a few dense matrices and leaves every
routed expert and Engram table byte-identical. The GGUF plan is shape-driven, so the
variant's GGUF has the base's exact layout and only the payloads of the edited dense
tensors differ. Instead of a second 466 GiB file, this tool downloads nothing itself: it
takes whichever variant shards are present, re-encodes their dense tensors, writes them
at their offsets in the base GGUF, and records the previous bytes in an undo file so the
file flips back with --revert. Experts in the present shards are checked to be identical
to the GGUF first; a differing expert aborts before any write.

Shards that are not present are described by `<shard>.header.json` sidecars
({"file_size", "header_length", "header"}) so the complete plan can be built;
--prepare fetches those sidecars and lists the shards whose LFS hash differs from the
base repository.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import (SourceDB, NativeQuantizer, Imatrix, build_plan, check_payload,
                                 item_sources, validate_scales, scale_name, QUANTIZATION)
from glm53_manifest import DTYPE_BYTES, checked_product, load_index, load_safetensors_header
from glm53_quantize import (GGUF_STRING, align, read_exact, read_u32, read_u64,
                            read_gguf_string, skip_gguf_value)


def sidecar_path(hf_dir, shard):
    return os.path.join(hf_dir, shard + ".header.json")


def load_sidecar(path):
    document = json.loads(Path(path).read_text())
    header_length, file_size = document["header_length"], document["file_size"]
    tensors = {}
    for name, entry in document["header"].items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        nbytes = checked_product(entry["shape"], name) * DTYPE_BYTES[entry["dtype"]]
        if end - start != nbytes or 8 + header_length + end > file_size:
            raise ValueError(f"{path}: inconsistent sidecar entry for {name}")
        tensors[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                         "offset": 8 + header_length + start, "nbytes": nbytes}
    return tensors


class PartialSourceDB(SourceDB):
    """A SourceDB whose absent shards are known only through header sidecars."""

    def __init__(self, hf_dir, support=False):
        import threading
        self.hf_dir = hf_dir
        document, self.weight_map = load_index(os.path.join(hf_dir, "model.safetensors.index.json"))
        self.declared_bytes = document.get("metadata", {}).get("total_size")
        self.tensors, self._fds, self._fd_lock = {}, {}, threading.Lock()
        self.present = set()
        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(hf_dir, shard)
            if os.path.isfile(path):
                loaded = load_safetensors_header(path)
                self.present.add(shard)
            elif os.path.isfile(sidecar_path(hf_dir, shard)):
                loaded = load_sidecar(sidecar_path(hf_dir, shard))
            else:
                raise ValueError(f"neither {shard} nor its header sidecar is present")
            for name, info in loaded.items():
                if self.weight_map.get(name) != shard:
                    raise ValueError(f"index assigns {name} to {self.weight_map.get(name)!r}, not {shard}")
                self.tensors[name] = dict(info, shard=shard)
        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            raise ValueError(f"source headers are incomplete; first missing tensor is {missing[0]}")
        validate_scales(self.tensors, support)

    def read(self, name):
        if self.info(name)["shard"] not in self.present:
            raise ValueError(f"{name}: its shard is not present")
        return super().read(name)


def open_gguf(fp, plan):
    """Walk the header, confirm the plan's layout, return (data_start, revision_value_offset)."""
    if read_exact(fp, 4, "magic") != b"GGUF" or read_u32(fp, "version") != 3:
        raise ValueError("expected GGUF v3")
    if read_u64(fp, "tensor count") != len(plan):
        raise ValueError("tensor count differs from the plan")
    revision_at = None
    for _ in range(read_u64(fp, "metadata count")):
        key = read_gguf_string(fp, "metadata key")
        kind = read_u32(fp, "metadata type")
        if key == "general.source.revision":
            if kind != GGUF_STRING:
                raise ValueError("revision is not a string")
            length = read_u64(fp, "revision length")
            revision_at = (fp.tell(), length)
            fp.seek(length, os.SEEK_CUR)
        else:
            skip_gguf_value(fp, kind)
    for item in plan:
        name = read_gguf_string(fp, "tensor name")
        rank = read_u32(fp, "tensor rank")
        shape = tuple(read_u64(fp, "dimension") for _ in range(rank))
        kind, offset = read_u32(fp, "tensor type"), read_u64(fp, "tensor offset")
        if (name, shape, kind, offset) != (item.name, item.shape, item.qtype, item.offset):
            raise ValueError(f"{item.name}: GGUF layout differs from the plan")
    if revision_at is None:
        raise ValueError("GGUF has no general.source.revision")
    start = align(fp.tell(), GGUF_ALIGNMENT)
    if os.fstat(fp.fileno()).st_size != start + plan[-1].offset + align(plan[-1].nbytes, GGUF_ALIGNMENT):
        raise ValueError("GGUF size differs from the plan")
    return start, revision_at


def undo_record(name, offset, data):
    encoded = name.encode()
    return struct.pack("<I", len(encoded)) + encoded + struct.pack("<QQ", offset, len(data)) + data


def read_undo(path):
    with open(path, "rb") as fp:
        while True:
            head = fp.read(4)
            if not head:
                return
            name = read_exact(fp, struct.unpack("<I", head)[0], "undo name").decode()
            offset, nbytes = struct.unpack("<QQ", read_exact(fp, 16, "undo span"))
            yield name, offset, read_exact(fp, nbytes, name)


def patch(args):
    config = json.loads((Path(args.hf) / "config.json").read_text())
    db = PartialSourceDB(args.hf, args.dspark_support)
    shards = set(args.shards) if args.shards else set(db.present)
    if shards - db.present:
        raise ValueError(f"requested shards are not present: {sorted(shards - db.present)}")
    plan = build_plan(db, config, args.quant, args.dspark_support)
    quantizer = NativeQuantizer(args.quants_library)
    imatrix = Imatrix(None, quantizer.np)
    changed, checked_experts, written = [], 0, 0
    with open(args.gguf, "r+b") as fp, open(args.undo, "ab") as undo:
        start, (revision_at, revision_length) = open_gguf(fp, plan)
        for item in plan:
            item_shards = {db.info(name)["shard"] for name in item_sources(item)}
            if not item_shards & shards:
                continue
            if item.role == "engram_disk":
                raise ValueError(f"{item.name}: Engram shards must not be patched; they are identical by hash")
            if item.is_expert:
                # Sampled byte comparison against the variant: a differing expert means
                # this variant is not a dense-only edit and the whole approach is wrong.
                check_payload(fp, start + item.offset, item, db, quantizer, imatrix)
                checked_experts += 1
                continue
            expected = quantizer.encode(quantizer.to_f32(db, item.source), item.qtype)
            if len(expected) != item.nbytes:
                raise ValueError(f"{item.name}: encoded size differs from the plan")
            fp.seek(start + item.offset)
            current = read_exact(fp, item.nbytes, item.name)
            if current == expected:
                continue
            undo.write(undo_record(item.name, start + item.offset, current))
            undo.flush()
            os.fsync(undo.fileno())
            fp.seek(start + item.offset)
            fp.write(expected)
            fp.flush()
            os.fsync(fp.fileno())
            changed.append(item.name)
            written += item.nbytes
            print(f"patched {item.name} ({item.nbytes} bytes)", flush=True)
        fp.seek(revision_at)
        current = read_exact(fp, revision_length, "revision")
        if current != args.source_revision.encode():
            if revision_length != len(args.source_revision):
                raise ValueError("revision strings differ in length; cannot rewrite in place")
            undo.write(undo_record("general.source.revision", revision_at, current))
            fp.seek(revision_at)
            fp.write(args.source_revision.encode())
            fp.flush()
            os.fsync(fp.fileno())
    manifest = Path(args.undo + ".json")
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    entries.append({"gguf": os.path.abspath(args.gguf), "shards": sorted(shards),
                    "variant_revision": args.source_revision, "changed": changed,
                    "experts_checked": checked_experts, "bytes_written": written})
    manifest.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"shards {sorted(shards)}: {len(changed)} dense tensors rewritten, {written} bytes, "
          f"{checked_experts} expert tensors identical", flush=True)


def revert(args):
    count = 0
    with open(args.gguf, "r+b") as fp:
        for name, offset, data in read_undo(args.undo):
            fp.seek(offset)
            fp.write(data)
            count += 1
        fp.flush()
        os.fsync(fp.fileno())
    print(f"restored {count} spans from {args.undo}", flush=True)


def prepare(args):
    """Header sidecars for every shard of the variant, and the list of shards whose LFS
    hash differs from the base repository at the pinned revisions."""
    from huggingface_hub import HfApi, HfFileSystem
    api, fs = HfApi(), HfFileSystem()
    base = {s.rfilename: s.lfs.sha256 for s in api.model_info(
        args.base_repo, revision=args.base_revision, files_metadata=True).siblings if s.lfs}
    variant = api.model_info(args.variant_repo, revision=args.source_revision, files_metadata=True)
    differing = []
    for sibling in sorted(variant.siblings, key=lambda s: s.rfilename):
        name = sibling.rfilename
        if not name.endswith(".safetensors"):
            continue
        if base.get(name) != sibling.lfs.sha256:
            differing.append(name)
        target = sidecar_path(args.hf, name)
        if os.path.exists(target):
            continue
        with fs.open(f"{args.variant_repo}/{name}", "rb", revision=args.source_revision) as fp:
            header_length = struct.unpack("<Q", fp.read(8))[0]
            header = json.loads(fp.read(header_length))
        Path(target).write_text(json.dumps({"file_size": sibling.size, "header_length": header_length,
                                            "header": header}))
        print(f"sidecar {name}", flush=True)
    Path(args.hf, "differing-shards.txt").write_text("".join(f"{n}\n" for n in differing))
    print(f"{len(differing)} shards differ from {args.base_repo}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="variant checkpoint dir: config, index, shards or sidecars")
    parser.add_argument("--source-revision", required=True, help="variant commit hash")
    parser.add_argument("--gguf")
    parser.add_argument("--quant", choices=QUANTIZATION, default="mxfp4")
    parser.add_argument("--undo", help="undo file; appended by --patch, replayed by --revert")
    parser.add_argument("--shards", nargs="*", help="shards to patch (default: every present shard)")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dspark-support", action="store_true", help="the GGUF is a DSpark support model")
    parser.add_argument("--base-repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    parser.add_argument("--base-revision", default="dba1be0a40aa45a94ad051997016db3960a90277")
    parser.add_argument("--variant-repo")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library", default=str(Path(__file__).with_name(f"libds4quants.{suffix}")))
    args = parser.parse_args()
    if args.prepare:
        if not args.variant_repo:
            parser.error("--prepare needs --variant-repo")
        return prepare(args)
    if not args.gguf or not args.undo:
        parser.error("--gguf and --undo are required")
    return revert(args) if args.revert else patch(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(f"deepseek41-patch: {error}")
