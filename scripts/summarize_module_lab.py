#!/usr/bin/env python3
"""Summarize completed development runs; never selects a TEST or drops a seed."""
import argparse
import json
from pathlib import Path
from statistics import mean

PHASES = ['screen', 'followup', 'replication', 'gnn_ln_replication', 'bc_kl',
          'public_pilot', 'public_scaled']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    summaries = {p: json.loads((Path(args.runs) / f'module_lab_{p}_20261019' / 'summary.json').read_text()) for p in PHASES}
    phases = []
    for name, s in summaries.items():
        assert not s['TEST_evaluated']
        assert s['simulator_counts']['complete_decodes'] == s['simulator_counts']['validated_schedules']
        phases.append({'phase': name, 'executed_arms': len(s['results']),
                       'training_complete_candidates': sum(r.get('actual_complete_candidates', 0) for r in s['results']),
                       'simulator_counts': s['simulator_counts'],
                       'code_commit': json.loads((Path(args.runs) / f'module_lab_{name}_20261019' / 'metadata.json').read_text())['code_commit_at_start']})
    # These phases repeat seed11 BC/PPO controls exactly; count work, not extra evidence.
    grouped = {}
    for arch, phase in [('edge_attention', 'replication'), ('edge_gnn_ln', 'gnn_ln_replication')]:
        rows = [r for p in ['followup', phase] for r in summaries[p]['results'] if r['arm'] == arch]
        grouped[arch] = {}
        for objective in ['BC', 'ppo', 'sil', 'pairwise', 'bopo']:
            selected = sorted([r for r in rows if r['objective'] == objective], key=lambda r: r['seed'])
            assert [r['seed'] for r in selected] == [11, 22, 33]
            grouped[arch][objective] = {
                'seeds': [r['seed'] for r in selected],
                'mean_C_over_HEFT': mean(r['full_known_VAL_C_over_HEFT'] for r in selected),
                'per_seed_C_over_HEFT': [r['full_known_VAL_C_over_HEFT'] for r in selected],
                'per_seed_paired_gain_vs_own_BC_percent': [r.get('paired_vs_own_BC', {}).get('mean_paired_gain_percent') for r in selected],
                'per_seed_graph_CI95_percent': [r.get('paired_vs_own_BC', {}).get('graph_cluster_CI95_percent') for r in selected]}
    for objective in ['ppo_kl1', 'ppo_kl10']:
        rows = sorted([r for r in summaries['bc_kl']['results'] if r['objective'] == objective], key=lambda r: r['seed'])
        grouped['edge_attention'][objective] = {'seeds': [r['seed'] for r in rows],
            'mean_C_over_HEFT': mean(r['full_known_VAL_C_over_HEFT'] for r in rows),
            'mean_paired_gain_vs_own_BC_percent': mean(r['paired_vs_own_BC']['mean_paired_gain_percent'] for r in rows)}
    before, after = summaries['public_pilot'], summaries['public_scaled']
    # Timing/bookkeeping may differ. Every physical baseline objective must match.
    baselines = ['Greedy', 'RandomMean5', 'HEFT', 'PEFT', 'MSLBL', 'Portfolio', 'Teacher64']
    first = {r['id']: r for r in before['reference_rows']}
    second = {r['id']: r for r in after['reference_rows']}
    assert set(first) == set(second)
    assert all(first[k][b] == second[k][b] for k in first for b in baselines)
    public = {'baseline_C_over_HEFT': {b: mean(r[b] / r['HEFT'] for r in second.values()) for b in baselines},
              'exact_matched_baseline_objectives': len(first) * len(baselines),
              'normalized_EFT_checks': after['simulator_counts'],
              'rows': []}
    for phase in ['public_pilot', 'public_scaled']:
        for r in summaries[phase]['results']:
            public['rows'].append({k: r[k] for k in ['arm', 'seed', 'objective', 'selected_step', 'full_known_VAL_C_over_HEFT',
                                                    'last_full_known_VAL_C_over_HEFT', 'paired_vs_own_BC', 'last_paired_vs_own_BC',
                                                    'strong_control_comparisons'] if k in r} | {'phase': phase})
    report = {'classification': 'EXPLORATORY development only; checkpoint-selection bias not included in intervals',
              'phases': phases, 'executed_arms_including_control_repeats': sum(p['executed_arms'] for p in phases),
              'exact_control_repeat_arms': 6,
              'training_complete_candidates_including_control_repeats': sum(p['training_complete_candidates'] for p in phases),
              'synthetic_grouped_results': grouped, 'public': public,
              'TEST_evaluated': False, 'frozen_default_promoted': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['executed_arms_including_control_repeats', 'exact_control_repeat_arms',
                                           'training_complete_candidates_including_control_repeats', 'TEST_evaluated']}))


if __name__ == '__main__':
    main()
