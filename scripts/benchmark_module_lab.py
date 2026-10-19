#!/usr/bin/env python3
"""Cold per-instance VAL inference timing including preparation and JSON RPC.

Model/dataset startup and one TRAIN-graph warmup are explicitly excluded. This is
an engineering benchmark, not a fresh quality test or an optimized GPU serving test.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.module_lab.models import ModuleActor, tensors
from scripts.run_module_lab import Simulation, choose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', default=str(ROOT / 'data/reference_dataset.jsonl.gz'))
    ap.add_argument('--sim-python', default=sys.executable)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit', type=int, default=24)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ModuleActor(saved['arm'], saved['hidden'], saved['layers']).to(args.device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    sim = Simulation(args.sim_python, args.data, decoder=saved.get('decoder', 'legacy'))
    sync = lambda: torch.cuda.synchronize() if torch.device(args.device).type == 'cuda' else None
    try:
        warm_id = choose(sim.info['records'], 'train', 1, 17)[0]
        warm = sim.call(command='get', ids=[warm_id])[0]
        with torch.no_grad():
            model(tensors(warm['graph'], args.device))
            sync()
        if torch.device(args.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        rows = []
        ids = choose(sim.info['records'], 'val', args.limit, 1903)
        for key in ids:
            sync()
            start = time.perf_counter()
            g = sim.call(command='get', ids=[key])[0]
            x = tensors(g['graph'], args.device)
            sync()
            prepared = time.perf_counter()
            before = model.forward_calls
            with torch.no_grad():
                logits = model(x)
                sync()
            forwarded = time.perf_counter()
            if model.forward_calls - before != 1:
                raise AssertionError('full forward count')
            r = sim.call(command='rollout', sample=False,
                         items=[{'id': key, 'logits': logits.cpu().tolist()}])[0]
            end = time.perf_counter()
            rows.append({'id': key, 'latent_id': g['latent_id'], 'makespan': r['makespan'],
                         'C_over_HEFT': r['makespan'] / g['reference'],
                         'prepare_and_RPC_seconds': prepared - start,
                         'forward_seconds': forwarded - prepared,
                         'output_copy_decode_and_RPC_seconds': end - forwarded,
                         'end_to_end_seconds': end - start, 'network_forwards': 1, 'complete_decodes': 1})
        fields = ['prepare_and_RPC_seconds', 'forward_seconds', 'output_copy_decode_and_RPC_seconds', 'end_to_end_seconds']
        result = {'scope': 'known VAL cold-instance engineering timing, not TEST',
                  'checkpoint': Path(args.checkpoint).name, 'device': args.device,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'warmup_forward_calls_excluded': 1, 'startup_and_dataset_load_excluded': True,
                  'rows': rows, 'means_seconds': {f: statistics.mean(r[f] for r in rows) for f in fields},
                  'median_end_to_end_seconds': statistics.median(r['end_to_end_seconds'] for r in rows),
                  'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.device(args.device).type == 'cuda' else None}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
