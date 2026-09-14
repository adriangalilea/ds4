#!/usr/bin/env python3
"""Build only the shared embedding/head for local DSpark replay; not a target model."""
import argparse
import hashlib
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from deepseek41_quantize import SourceDB, NativeQuantizer
from glm53_quantize import TensorPlan, align, kv_string, qtype_nbytes, tensor_header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head-sha256", required=True)
    args = parser.parse_args()
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=lambda _: None,
                  tensor_filter=lambda n: n in ("embed.weight", "head.weight"))
    quantizer = NativeQuantizer(str(ROOT / "gguf-tools/libds4quants.dylib"))
    items = [TensorPlan("token_embd.weight", (5120, 129280), 1, "embedding", source="embed.weight"),
             TensorPlan("output.weight", (5120, 129280), 8, "output", source="head.weight")]
    offset = 0
    for item in items:
        assert db.info(item.source)["shape"] == [129280, 5120]
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        item.offset = offset
        offset += align(item.nbytes, 32)
    records = [kv_string("general.architecture", "deepseek41-shared-fixture"),
               kv_string("general.source.revision", "dba1be0a40aa45a94ad051997016db3960a90277")]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(items), len(records))
    header += b"".join(records) + b"".join(tensor_header(item) for item in items)
    header += bytes(align(len(header), 32) - len(header))
    with args.out.open("xb") as fp:
        fp.write(header)
        for item in items:
            digest = hashlib.sha256()
            for row in range(0, 129280, 2048):
                count = min(2048, 129280 - row)
                encoded = quantizer.encode(quantizer.to_f32(db, item.source, row, count), item.qtype)
                fp.write(encoded)
                digest.update(encoded)
            fp.write(bytes(align(item.nbytes, 32) - item.nbytes))
            print(item.name, digest.hexdigest(), flush=True)
            if item.name == "output.weight":
                assert digest.hexdigest() == args.head_sha256
    db.close()


if __name__ == "__main__":
    main()
