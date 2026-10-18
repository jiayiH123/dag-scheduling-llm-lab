#!/usr/bin/env python3
"""Prepare 24 previously inspected WfInstances configurations, not a blind test.

Downloads only when --download is explicit. Reads JSON as data; it never runs
workflow commands, unpickles files, or generates training teacher labels.
"""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.reference import core, graph_features
from daglab.workflow_trace import load_trace

MANIFEST = ROOT / "data/public_smoke_sources.json"


def checked_bytes(data, entry):
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"] or len(data) != entry["bytes"]:
        raise ValueError("source hash/size mismatch for " + entry["file"])
    raw = json.loads(data)
    if len(raw["workflow"]["specification"]["tasks"]) != entry["tasks"]:
        raise ValueError("source task-count mismatch for " + entry["file"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "runs/public_smoke_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/public_smoke.jsonl.gz")
    parser.add_argument("--download", action="store_true",
                        help="explicitly download missing pinned JSON sources")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuses overwrite: " + str(args.output))
    manifest = json.loads(MANIFEST.read_text())
    for entry in manifest["entries"]:
        path = args.raw_dir / entry["file"]
        if path.exists():
            checked_bytes(path.read_bytes(), entry)
            continue
        if not args.download:
            raise FileNotFoundError(str(path) + "; supply existing raw JSON files or pass --download")
        request = urllib.request.Request(entry["source_url"], headers={"User-Agent": "dag-scheduling-lab-public-smoke/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = checked_bytes(response.read(), entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A mismatching or existing source is never silently replaced.
        with path.open("xb") as stream:
            stream.write(data)

    records = []
    for entry, profile, eta in itertools.product(
            manifest["entries"], manifest["profiles"], manifest["budget_etas"]):
        inst = load_trace(args.raw_dir / entry["file"], entry["source"], profile,
                          eta, family=entry["family"])
        inst.update(source_url=entry["source_url"],
                    source_repository_commit=manifest["pinned_commit"])
        graph, _ = graph_features(inst)
        reference = core.schedule(inst, "HEFT")
        core.validate(inst, reference)
        records.append({
            "key": "val_public_smoke_" + entry["source"] + "_" + profile + "_" + inst["budget_mode"],
            "instance": inst, "graph": graph, "reference": reference["makespan"],
            "classification": manifest["classification"],
            "previously_inspected": True,
            "note": "Known smoke/development input, not a fresh held-out test; no teacher labels."
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "xt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "raw_dir": str(args.raw_dir),
        "configs": len(records),
        "latent_graphs": len({r["instance"]["latent_id"] for r in records}),
        "application_families": len({r["instance"]["family"] for r in records}),
        "classification": manifest["classification"],
        "source_commit": manifest["pinned_commit"],
        "teacher_labels_generated": False,
        "note": "All source hashes checked; all 24 HEFT reference schedules validated. Known discovery traces are not a fresh final test."
    }, indent=2))


if __name__ == "__main__":
    main()
