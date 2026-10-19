#!/usr/bin/env python3
"""Small text/JSON report for pull-only machines; no upload or model weights."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text())
    metadata = json.loads((run / 'metadata.json').read_text())
    selection = json.loads((run / 'data_selection.json').read_text())
    rows = []
    fields = ['arm', 'seed', 'objective', 'parameters', 'selected_step', 'full_known_VAL_C_over_HEFT',
              'last_full_known_VAL_C_over_HEFT', 'paired_vs_own_BC', 'last_paired_vs_own_BC',
              'actual_complete_candidates', 'optimizer_steps', 'training_seconds_including_validation']
    for result in summary['results']:
        row = {k: result[k] for k in fields if k in result}
        row['strong_control_mean_paired_gain_percent'] = {
            name: value['mean_paired_gain_percent'] for name, value in result['strong_control_comparisons'].items()}
        rows.append(row)
    result = {'scope': summary['scope'], 'code_commit': metadata['code_commit_at_start'],
              'protocol_commit': metadata['protocol_commit'], 'config': metadata['config'],
              'torch': metadata['torch'], 'device': metadata['device'],
              'data_sha256': selection['data_sha256'],
              'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              'selection_sha256': hashlib.sha256((run / 'data_selection.json').read_bytes()).hexdigest(),
              'selected_architectures': summary['selected_architectures'], 'results': rows,
              'simulator_counts': summary['simulator_counts'],
              'max_absolute_logprob_replay_error': summary['max_absolute_logprob_replay_error'],
              'TEST_evaluated': summary['TEST_evaluated'], 'deploy_forwards': 1, 'deploy_decodes': 1,
              'raw_files_remain_on_machine': ['summary.json', 'curves.jsonl', 'data_selection.json',
                                              'metadata.json', 'mechanism_checks.json'],
              'caveat': 'Copy this report; no GitHub push, credentials or model-weight upload needed.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(str(out))


if __name__ == '__main__':
    main()
