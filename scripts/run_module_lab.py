#!/usr/bin/env python3
"""Staged, reproducible module screen, CPU/CUDA, no TEST access.

The simulator can run in a separate Python environment. Outputs are additive and
refuse overwrite. This entry point is independent of any server-only pilot code.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.module_lab.models import ARMS, ModuleActor, conditionals, tensors


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Simulation:
    def __init__(self, executable, data):
        self.process = subprocess.Popen([executable, '-m', 'daglab.module_lab.simulator'], cwd=ROOT,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.info = self.call(command='init', data=str(Path(data).resolve()))

    def call(self, **query):
        self.process.stdin.write(json.dumps(query, allow_nan=False) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('simulation interpreter exited; see its stderr')
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            self.call(command='close')
            self.process.wait(timeout=20)


def choose(records, prefix, limit, seed):
    groups = {}
    for record in records:
        if record['id'].startswith(prefix):
            groups.setdefault(record['latent_id'], []).append(record['id'])
    rng = random.Random(seed)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    for group in groups.values():
        group.sort()
        rng.shuffle(group)
    result = []
    while len(result) < limit and any(groups.values()):
        for key in group_ids:
            if groups[key] and len(result) < limit:
                result.append(groups[key].pop())
    return result


def paired(values, baseline, lookup):
    groups = {}
    gains = []
    for key, value in values.items():
        gain = 100 * (baseline[key] - value) / baseline[key]
        gains.append(gain)
        groups.setdefault(lookup[key]['latent_id'], []).append(gain)
    grouped = list(groups.values())
    rng = random.Random(271828)
    bootstrap = []
    for _ in range(1000):
        draw = [grouped[rng.randrange(len(grouped))] for _ in grouped]
        bootstrap.append(sum(map(sum, draw)) / sum(map(len, draw)))
    bootstrap.sort()
    return {'mean_paired_gain_percent': sum(gains) / len(gains),
            'graph_cluster_CI95_percent': [bootstrap[25], bootstrap[974]],
            'wins': sum(g > 1e-9 for g in gains), 'losses': sum(g < -1e-9 for g in gains),
            'ties': sum(abs(g) <= 1e-9 for g in gains), 'worst_regression_percent': -min(gains),
            'configs': len(values), 'latent_graphs': len(groups),
            'scope': 'exploratory known VAL, checkpoint-selection uncertainty not included'}


class Lab:
    def __init__(self, args, config, out, simulation):
        self.args, self.cfg, self.out, self.sim = args, config, out, simulation
        self.device = torch.device(args.device)
        self.cache = {}
        self.lookup = {r['id']: r for r in simulation.info['records']}
        self.train = choose(simulation.info['records'], 'train', config['train_limit'], 1901)
        self.val = choose(simulation.info['records'], 'val', config['val_limit'], 1902)
        self.fullval = sorted(k for k in self.lookup if k.startswith('val'))
        self.probability_error = 0.
        self.optimizer_steps = 0
        self.curves = (out / 'curves.jsonl').open('w')
        write_json(out / 'data_selection.json', {'TRAIN': self.train, 'screen_VAL': self.val,
                    'report_VAL': self.fullval, 'data_sha256': simulation.info['data_sha256'],
                    'TEST_evaluated': False})

    def get(self, ids):
        missing = [k for k in ids if k not in self.cache]
        for start in range(0, len(missing), 20):
            for g in self.sim.call(command='get', ids=missing[start:start + 20]):
                g['tensor'] = tensors(g['graph'], self.device)
                self.cache[g['id']] = g
        return [self.cache[k] for k in ids]

    def emit(self, value):
        self.curves.write(json.dumps(value, allow_nan=False) + '\n')
        self.curves.flush()

    def evaluate(self, model, ids):
        values = {}
        model.eval()
        with torch.no_grad():
            for start in range(0, len(ids), 20):
                items = []
                for g in self.get(ids[start:start + 20]):
                    before = model.forward_calls
                    x = model(g['tensor'])
                    if model.forward_calls - before != 1:
                        raise AssertionError('deployment full-forward count')
                    items.append({'id': g['id'], 'logits': x.cpu().tolist()})
                for row in self.sim.call(command='rollout', sample=False, items=items):
                    values[row['id']] = row['makespan']
        model.train()
        return values

    def ratio(self, values):
        return sum(v / self.cache[k]['reference'] for k, v in values.items()) / len(values)

    def save(self, path, model, **metadata):
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'arm': model.arm, 'hidden': self.cfg['hidden'], 'layers': self.cfg['layers'],
                    **metadata}, path)

    def apply_gradient(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
        if not math.isfinite(norm):
            raise ValueError('nonfinite gradient')
        optimizer.step()
        self.optimizer_steps += 1
        return norm

    def audits(self):
        outputs = []
        ids = self.train[:2]
        for arm in self.cfg['architectures']:
            torch.manual_seed(0)
            model = ModuleActor(arm, self.cfg['hidden'], self.cfg['layers']).to(self.device)
            for g in self.get(ids):
                x = model(g['tensor'])
                zero = self.sim.call(command='rollout', sample=False,
                                     items=[{'id': g['id'], 'logits': x.detach().cpu().tolist()}])[0]
                if not ARMS[arm]['prior'] and zero['makespan'] != g['reference']:
                    raise AssertionError('zero head must reproduce HEFT makespan')
                # Nonzero a/w exercise both budget and offsets in own-prefix replay.
                probe = x.detach() + torch.linspace(-.7, .7, x.numel(), device=self.device).reshape_as(x)
                r = self.sim.call(command='rollout', sample=True,
                                  items=[{'id': g['id'], 'logits': probe.cpu().tolist(), 'seed': 189}])[0]
                self.check_probabilities(probe, r)
                outputs.append({'arm': arm, 'id': g['id'], 'initial_C_over_HEFT': zero['makespan'] / g['reference'],
                                'one_forward_verified': True, 'own_prefix_replay': True})
        # Overfit is a diagnostic of learnability/gradients, not a quality claim.
        overfit = []
        for arm in ['legacy_gnn', 'edge_attention']:
            torch.manual_seed(19)
            model = ModuleActor(arm, self.cfg['hidden'], self.cfg['layers']).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            g = self.get(self.train[:1])[0]
            losses = []
            for _ in range(81):
                logs, _ = conditionals(model(g['tensor']), g['teacher_decisions'])
                loss = -logs.mean()
                losses.append(float(loss.detach()))
                self.apply_gradient(loss, optimizer, model.parameters())
            overfit.append({'arm': arm, 'id': g['id'], 'initial_nll': losses[0],
                            'final_nll': losses[-1], 'diagnostic_steps': 81,
                            'reduced': losses[-1] < losses[0]})
        result = {'checks': outputs, 'one_batch_overfit': overfit,
                  'max_absolute_logprob_replay_error': self.probability_error,
                  'device': str(self.device), 'TEST_evaluated': False}
        write_json(self.out / 'mechanism_checks.json', result)
        return result

    def check_probabilities(self, logits, trajectory):
        with torch.no_grad():
            logs, _ = conditionals(logits, trajectory['decisions'])
            old = torch.tensor([d['selected_log_probability'] for d in trajectory['decisions']],
                               device=self.device, dtype=logs.dtype)
            error = float((logs - old).abs().max())
        self.probability_error = max(self.probability_error, error)
        if error > 5e-5:
            raise AssertionError(('own-prefix probability mismatch', error))

    def bc(self, arm, seed):
        torch.manual_seed(seed)
        model = ModuleActor(arm, self.cfg['hidden'], self.cfg['layers']).to(self.device)
        rng = random.Random(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg['bc_lr'])
        values = self.evaluate(model, self.val)
        best, best_step, best_state = self.ratio(values), 0, deepcopy(model.state_dict())
        initial = best
        prior_only = self.evaluate(model, self.fullval) if ARMS[arm]['prior'] else None
        start = time.perf_counter()
        for step in range(1, self.cfg['bc_updates'] + 1):
            batch = self.get(rng.sample(self.train, self.cfg['batch']))
            loss = torch.stack([-conditionals(model(g['tensor']), g['teacher_decisions'])[0].mean()
                                for g in batch]).mean()
            grad = self.apply_gradient(loss, optimizer, model.parameters())
            entry = {'arm': arm, 'seed': seed, 'stage': 'BC', 'step': step,
                     'loss': float(loss.detach()), 'grad_norm': grad}
            if step % self.cfg['eval_every'] == 0 or step == self.cfg['bc_updates']:
                value = self.ratio(self.evaluate(model, self.val))
                entry['screen_VAL_C_over_HEFT'] = value
                if value < best:
                    best, best_step, best_state = value, step, deepcopy(model.state_dict())
                print(f'BC {arm} seed{seed} step{step} val={value:.6f} best={best:.6f}', flush=True)
            self.emit(entry)
        self.save(self.out / f'{arm}_s{seed}_BC_last.pt', model, selected=False)
        model.load_state_dict(best_state)
        self.save(self.out / f'{arm}_s{seed}_BC.pt', model, selected_step=best_step)
        final = self.evaluate(model, self.fullval)
        detail = {'arm': arm, 'seed': seed, 'objective': 'BC', 'initial_screen_ratio': initial,
                  'selected_step': best_step, 'screen_VAL_C_over_HEFT': best,
                  'full_known_VAL_C_over_HEFT': self.ratio(final),
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'training_seconds_including_validation': time.perf_counter() - start,
                  'values': final, 'init': 'from_scratch_zero_output_heads'}
        if prior_only is not None:
            detail['analytic_prior_only_values'] = prior_only
            detail['analytic_prior_only_full_VAL_C_over_HEFT'] = self.ratio(prior_only)
        write_json(self.out / f'{arm}_s{seed}_BC.json', detail)
        return model, detail

    def loss(self, objective, logits, g, trajectories):
        if objective == 'ppo':
            terms, kls = [], []
            for r in trajectories[1:]:  # deterministic greedy is not on-policy sampling
                logs, entropy = conditionals(logits, r['decisions'])
                old = torch.tensor([d['selected_log_probability'] for d in r['decisions']],
                                   dtype=logs.dtype, device=self.device)
                delta = logs - old
                ratio = delta.exp()
                advantage = (trajectories[0]['makespan'] - r['makespan']) / g['reference']
                terms.append(-torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).sum()
                             - self.cfg['entropy'] * entropy.sum())
                kls.append(((ratio - 1) - delta).mean().detach())
            return torch.stack(terms).mean(), float(torch.stack(kls).mean())
        ordered = sorted(trajectories, key=lambda r: r['makespan'])
        unique, seen = [], set()
        for r in ordered:
            mapping = tuple(d['action'] for d in r['decisions'])
            if mapping not in seen:
                seen.add(mapping)
                unique.append(r)
        winner = unique[0]
        lw = conditionals(logits, winner['decisions'])[0].mean()
        if objective == 'sil':
            return (-lw if winner['makespan'] < trajectories[0]['makespan'] else lw * 0), 0.
        if len(unique) > self.cfg['retained']:
            indices = [round(i * (len(unique) - 1) / (self.cfg['retained'] - 1))
                       for i in range(self.cfg['retained'])]
            unique = [unique[i] for i in indices]
        pairs = []
        for loser in unique[1:]:
            if loser['makespan'] <= winner['makespan']:
                continue  # never train an arbitrary preference for objective ties
            ll = conditionals(logits, loser['decisions'])[0].mean()
            factor = loser['makespan'] / winner['makespan'] if objective == 'bopo' else 1.
            pairs.append(F.softplus(factor * (ll - lw)))
        return (torch.stack(pairs).mean() if pairs else lw * 0), 0.

    def optimize(self, bc, bc_detail, objective, seed):
        model = deepcopy(bc)
        arm = model.arm
        rng = random.Random(seed + 320)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg['rl_lr'])
        values = self.evaluate(model, self.val)
        best, best_step, state = self.ratio(values), 0, deepcopy(model.state_dict())
        start, sampled, steps = time.perf_counter(), 0, 0
        signal = []
        for update in range(1, self.cfg['rl_updates'] + 1):
            batch = self.get(rng.sample(self.train, self.cfg['batch']))
            with torch.no_grad():
                xs = [model(g['tensor']) for g in batch]
            greedy_items = [{'id': g['id'], 'logits': x.cpu().tolist()} for g, x in zip(batch, xs)]
            greedy = self.sim.call(command='rollout', sample=False, items=greedy_items)
            sample_items = []
            for j, (g, x) in enumerate(zip(batch, xs)):
                sample_items.extend({'id': g['id'], 'logits': x.cpu().tolist(),
                                     'seed': seed * 10000000 + update * 10000 + j * 100 + candidate}
                                    for candidate in range(self.cfg['candidates'] - 1))
            randoms = self.sim.call(command='rollout', sample=True, items=sample_items)
            trajectories = [[greedy[j]] + randoms[j * (self.cfg['candidates'] - 1):
                                                (j + 1) * (self.cfg['candidates'] - 1)] for j in range(len(batch))]
            sampled += len(batch) * self.cfg['candidates']
            for g, x, rs in zip(batch, xs, trajectories):
                for r in rs:
                    self.check_probabilities(x, r)
                signal.append({'step': update, 'id': g['id'], 'greedy': rs[0]['makespan'],
                               'best_candidate': min(r['makespan'] for r in rs),
                               'unique_mappings': len({tuple(d['action'] for d in r['decisions']) for r in rs}),
                               'candidates': len(rs)})
            epochs = 4 if objective == 'ppo' else 1
            for epoch in range(epochs):
                losses, main_losses, anchors, kls = [], [], [], []
                for g, rs in zip(batch, trajectories):
                    x = model(g['tensor'])
                    primary, kl = self.loss(objective, x, g, rs)
                    anchor = -conditionals(x, g['teacher_decisions'])[0].mean()
                    main_losses.append(primary)
                    anchors.append(anchor)
                    losses.append(primary + self.cfg['teacher_anchor'] * anchor)
                    kls.append(kl)
                if objective == 'ppo' and sum(kls) / len(kls) > .03:
                    break
                loss = torch.stack(losses).mean()
                grad = self.apply_gradient(loss, optimizer, model.parameters())
                steps += 1
            entry = {'arm': arm, 'seed': seed, 'stage': objective, 'step': update,
                     'loss': float(loss.detach()), 'primary_loss': float(torch.stack(main_losses).mean().detach()),
                     'anchor_nll': float(torch.stack(anchors).mean().detach()), 'grad_norm': grad,
                     'actual_complete_candidates_cumulative': sampled, 'optimizer_steps_cumulative': steps}
            if update % self.cfg['rl_eval_every'] == 0 or update == self.cfg['rl_updates']:
                value = self.ratio(self.evaluate(model, self.val))
                entry['screen_VAL_C_over_HEFT'] = value
                if value < best:
                    best, best_step, state = value, update, deepcopy(model.state_dict())
                print(f'{objective} {arm} seed{seed} step{update} val={value:.6f} best={best:.6f}', flush=True)
            self.emit(entry)
        last = self.evaluate(model, self.fullval)
        self.save(self.out / f'{arm}_s{seed}_{objective}_last.pt', model, selected=False)
        model.load_state_dict(state)
        self.save(self.out / f'{arm}_s{seed}_{objective}.pt', model, selected_step=best_step)
        final = self.evaluate(model, self.fullval)
        detail = {'arm': arm, 'seed': seed, 'objective': objective, 'selected_step': best_step,
                  'screen_VAL_C_over_HEFT': best, 'full_known_VAL_C_over_HEFT': self.ratio(final),
                  'last_full_known_VAL_C_over_HEFT': self.ratio(last),
                  'paired_vs_own_BC': paired(final, bc_detail['values'], self.lookup),
                  'last_paired_vs_own_BC': paired(last, bc_detail['values'], self.lookup),
                  'actual_complete_candidates': sampled, 'optimizer_steps': steps,
                  'training_seconds_including_validation': time.perf_counter() - start,
                  'values': final, 'last_values': last, 'candidate_diagnostics': signal}
        write_json(self.out / f'{arm}_s{seed}_{objective}.json', detail)
        return detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs/module_lab_local.json'))
    ap.add_argument('--data', default=str(ROOT / 'data/reference_dataset.jsonl.gz'))
    ap.add_argument('--sim-python', default=sys.executable)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output', required=True)
    ap.add_argument('--checks-only', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg['candidates'] < 2 or cfg['retained'] < 2:
        raise ValueError('at least2 actual candidates and retained entries')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    simulation = None
    try:
        simulation = Simulation(args.sim_python, args.data)
        lab = Lab(args, cfg, out, simulation)
        metadata = {'config': cfg, 'argv': sys.argv, 'torch': torch.__version__, 'device': args.device,
                    'code_commit_at_start': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'protocol_commit': '836ab82', 'classification': 'EXPLORATORY known synthetic TRAIN/VAL only',
                    'TEST_evaluated': False, 'historical_warmstart_used': False}
        write_json(out / 'metadata.json', metadata)
        checks = lab.audits()
        print(json.dumps({'checks_complete': True, 'overfit': checks['one_batch_overfit']}), flush=True)
        if args.checks_only:
            return
        bc_models, results = {}, []
        for seed in cfg['seeds']:
            for arm in cfg['architectures']:
                model, detail = lab.bc(arm, seed)
                bc_models[arm, seed] = (model, detail)
                results.append(detail)
        ranking = sorted(cfg['architectures'], key=lambda arm: sum(
            bc_models[arm, seed][1]['screen_VAL_C_over_HEFT'] for seed in cfg['seeds']) / len(cfg['seeds']))
        selected = ranking[:cfg['top_architectures']]
        write_json(out / 'architecture_selection.json', {'ranking': ranking, 'selected': selected,
                    'selection': 'mean screen VAL over configured seeds; no TEST', 'parameter_matched': False})
        for seed in cfg['seeds']:
            for arm in selected:
                model, detail = bc_models[arm, seed]
                for objective in cfg['objectives']:
                    result = lab.optimize(model, detail, objective, seed)
                    results.append(result)
                    write_json(out / 'partial_results.json', results)
        references = []
        for start in range(0, len(lab.fullval), 12):
            references.extend(simulation.call(command='references', ids=lab.fullval[start:start + 12]))
            print(f'controls {min(start + 12, len(lab.fullval))}/{len(lab.fullval)}', flush=True)
        controls = {name: {r['id']: r[name] for r in references}
                    for name in ['Greedy', 'HEFT', 'PEFT', 'MSLBL', 'Portfolio', 'Teacher64']}
        prior_result = next((r for r in results if 'analytic_prior_only_values' in r), None)
        if prior_result is not None:
            controls['BoundedOCTPriorOnly'] = prior_result['analytic_prior_only_values']
        for result in results:
            result['strong_control_comparisons'] = {name: paired(result['values'], values, lab.lookup)
                                                    for name, values in controls.items()}
        summary = {'scope': metadata['classification'], 'selected_architectures': selected,
                   'results': results, 'reference_rows': references, 'mechanism_checks': checks,
                   'simulator_counts': simulation.call(command='counts'),
                   'max_absolute_logprob_replay_error': lab.probability_error,
                   'total_optimizer_steps_including_overfit': lab.optimizer_steps,
                   'deploy_full_forwards': 1, 'deploy_decodes': 1, 'TEST_evaluated': False,
                   'caveat': 'one or development seeds, known VAL and adaptive architecture selection; not confirmation'}
        write_json(out / 'summary.json', summary)
        print(json.dumps({'output': str(out), 'selected_architectures': selected,
                          'results': [{k: v for k, v in r.items() if k in
                           ['arm', 'seed', 'objective', 'selected_step', 'full_known_VAL_C_over_HEFT', 'paired_vs_own_BC']}
                                      for r in results]}, indent=2), flush=True)
    except Exception as error:
        write_json(out / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if simulation:
            simulation.close()


if __name__ == '__main__':
    main()
