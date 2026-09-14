"""Replay complete captured target streams through the standalone Metal drafter."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--studio-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.dump / "meta.json").read_text())
    assert (meta["rows"], meta["dim"], meta["hc"], meta["first_generation_row"]) == (383, 5120, 4, 127)
    seeds = np.asarray(meta["generated"], dtype="<i4")
    assert seeds.shape == (256,)
    seed_path = args.output / "seeds.i32"
    seeds.tofile(seed_path)
    raw = np.fromfile(args.dump / "raw_hc.f32", dtype="<f4").reshape(383, 3, 4, 5120)
    means = np.fromfile(args.dump / "mean.f32", dtype="<f4").reshape(383, 3, 5120)
    # Equal-weight HC mean with round-to-nearest-even BF16 storage.
    average = raw.mean(axis=2)
    bits = average.view(np.uint32)
    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
    np.testing.assert_array_equal(rounded, means)
    assert np.isfinite(raw).all() and np.std(means) > 0
    oracle = {}
    for line in args.studio_log.read_text().splitlines():
        match = re.match(r"draft position=(\d+) matched=\d+ ids=(.*)", line)
        if match:
            oracle[int(match[1])] = [int(v.split(":")[0]) for v in match[2].split(",")]
    assert len(oracle) == 251
    results = {"capture_mean_exact": True, "rows": 383, "experiments": {}}
    for name, stream, mode in [("mean", "mean", "real"), ("zero", "mean", "zero"),
                               ("constant", "mean", "constant"),
                               ("collapsed", "collapsed", "real"),
                               ("normalized", "normalized", "real")]:
        command = [str(args.binary), "--replay", str(args.fixture), str(args.support),
                   str(args.dump / f"{stream}.f32"), str(seed_path), str(meta["start"]), mode]
        with (args.output / f"{name}.jsonl").open("w") as out, (args.output / f"{name}.err").open("w") as err:
            subprocess.run(command, stdout=out, stderr=err, check=True)
        rows = [json.loads(line) for line in (args.output / f"{name}.jsonl").read_text().splitlines()
                if line.startswith("{")]
        assert len(rows) == 256 and [row["seed"] for row in rows] == seeds.tolist()
        proposals = [row["proposals"] for row in rows]
        identical = sum(proposals[i] == expected for i, expected in oracle.items())
        if name == "mean":
            assert identical == 251, f"Replay differs from Studio at {251-identical} positions; ablations stopped"
        first = sum(proposals[i][0] == seeds[i + 1] for i in range(255))
        hist = [0] * 6
        for i in range(251):
            n = 0
            while n < 5 and proposals[i][n] == seeds[i + n + 1]:
                n += 1
            hist[n] += 1
        result = {"first_matches": first, "first_opportunities": 255,
                  "first_agreement": first / 255, "prefix_histogram": hist,
                  "mean_prefix": sum(i * n for i, n in enumerate(hist)) / 251,
                  "studio_identical_proposal_sequences": identical}
        results["experiments"][name] = result
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
