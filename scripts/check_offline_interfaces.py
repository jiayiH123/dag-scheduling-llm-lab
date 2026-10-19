#!/usr/bin/env python3
"""Bounded CPU data/forward/decoder engineering check, no neural optimization."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_module_lab import Simulation
from daglab.module_lab.models import ModuleActor, conditionals, set_probability_dtype, tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--sim-python', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--largest', action='store_true', help='exercise largest selected TRAIN and VAL graph')
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1); set_probability_dtype('float64')
    checks, simulators = [], []
    max_error = 0.
    for data in args.data:
        sim = Simulation(args.sim_python, data, 'normalized_eft')
        try:
            records = sim.info['records']
            chosen = [(max([r for r in records if r['id'].startswith(role)], key=lambda r: r['n'])['id']
                       if args.largest else next(r['id'] for r in records if r['id'].startswith(role)))
                      for role in ['train', 'val']]
            for g in sim.call(command='get', ids=chosen):
                for arm in ['edge_gnn_ln', 'edge_attention']:
                    torch.manual_seed(11)
                    model = ModuleActor(arm).eval()
                    with torch.no_grad():
                        output = model(tensors(g['graph'], torch.device('cpu')))
                    if model.forward_calls != 1:
                        raise AssertionError('not exactly one whole neural forward')
                    row = sim.call(command='rollout', items=[{'id': g['id'], 'logits': output.tolist()}], keep_schedule=True)[0]
                    if row['makespan'] != g['reference']:
                        raise AssertionError('zero-head HEFT identity')
                    logs, _ = conditionals(output, row['decisions'])
                    expected = torch.tensor([d['selected_log_probability'] for d in row['decisions']], dtype=torch.float64)
                    error = float((logs - expected).abs().max())
                    max_error = max(error, max_error)
                    if error > 1e-10:
                        raise AssertionError(('own-prefix probability replay', error))
                    checks.append({'data': str(Path(data).name), 'id': g['id'], 'arm': arm,
                                   'n': g['graph']['n'], 'full_forward_calls': model.forward_calls,
                                   'physical_validation': True, 'zero_head_HEFT_identity': True,
                                   'max_logprob_replay_error': error})
            simulators.append(sim.call(command='counts'))
        finally:
            sim.close()
    result = {'status': 'PASS', 'checks': checks, 'simulator_counts': simulators,
              'max_absolute_logprob_replay_error': max_error, 'neural_optimizer_steps': 0,
              'TEST_evaluated': False, 'CUDA_tested': False,
              'scope': 'CPU engineering on selected TRAIN/VAL inputs, random models with zero output heads; no policy quality claim'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['checks', 'simulator_counts']}))


if __name__ == '__main__':
    main()
