#!/usr/bin/env python3
"""Validate the DSpark layout against published safetensors headers, without payloads."""
import argparse
import copy
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools"))
from deepseek41_metadata import dspark_metadata
from deepseek41_quantize import build_plan, validate_scales
from glm53_quantize import QTYPE_F16, QTYPE_F32, QTYPE_Q4_K, QTYPE_Q8_0


def check(hf_dir, headers):
    config, records = dspark_metadata(hf_dir, "0" * 40)
    tensors = json.loads(Path(headers).read_text())
    db = types.SimpleNamespace(tensors=tensors, info=tensors.__getitem__)
    validate_scales(tensors, support=True)
    plan = build_plan(db, config, "q4", support=True)
    outputs = {item.name: item for item in plan}
    assert len(outputs) == len(plan) == 84
    assert all(item.name.startswith("mtp.") for item in plan)
    assert sum(item.is_expert for item in plan) == 9
    for item in plan:
        if item.is_expert:
            assert item.expert_count == 128 and item.qtype == QTYPE_Q4_K
    for name, shape, qt in (
        ("mtp.0.main_proj.weight", (15360, 5120), QTYPE_Q8_0),
        ("mtp.2.markov_head.markov_w1.weight", (256, 129280), QTYPE_F16),
        ("mtp.2.markov_head.markov_w2.weight", (256, 129280), QTYPE_Q8_0),
        ("mtp.2.confidence_head.proj.weight", (5376, 1), QTYPE_F32),
    ):
        assert (outputs[name].shape, outputs[name].qtype) == (shape, qt), name

    # Equal element count does not make the transposed projection compatible.
    bad = copy.deepcopy(tensors)
    bad["mtp.0.main_proj.weight"]["shape"].reverse()
    try:
        build_plan(types.SimpleNamespace(tensors=bad, info=bad.__getitem__), config, "q4", support=True)
    except ValueError:
        pass
    else:
        raise AssertionError("transposed main projection accepted")
    bad = dict(tensors, **{"mtp.0.unknown.weight": {"shape": [1], "dtype": "F32"}})
    try:
        build_plan(types.SimpleNamespace(tensors=bad, info=bad.__getitem__), config, "q4", support=True)
    except ValueError:
        pass
    else:
        raise AssertionError("unclaimed drafter tensor accepted")
    print(json.dumps({"source_tensors": len(tensors), "gguf_tensors": len(plan),
                      "payload_bytes": plan[-1].offset + plan[-1].nbytes,
                      "metadata_records": len(records), "layout": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--source-headers", required=True)
    args = parser.parse_args()
    check(args.hf_dir, args.source_headers)
