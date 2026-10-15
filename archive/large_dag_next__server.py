#!/usr/bin/env python3
"""Disjoint latent-DAG splits and simulator RPC for the larger-DAG pilot."""
from copy import deepcopy
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'experiments/budget_ppo'))
import server as previous
import budget_core as core
from cached_budget_decoder import CachedBudgetDecoder
from search_controls import portfolio, mapping_of

FAMILIES = ['pipeline', 'forkjoin', 'costly_downstream', 'random_packing']


def base(family, n, resource, seed, ccr):
    latent = f'{family}:{n}:{resource}:{seed}'
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(latent.encode()).digest()[:8], 'little'))
    work = (4*rng.integers(5, 61, n)).tolist()
    if family == 'pipeline':
        width = 4
        edges = [[v, v+width] for v in range(n-width)]
        edges += [[v, v+width+1] for v in range(n-width-1) if rng.random() < .25]
    elif family == 'forkjoin':
        edges = [[0, v] for v in range(1, n-1)]+[[v, n-1] for v in range(1, n-1)]
    elif family == 'costly_downstream':
        groups = [list(range(0, n-1, 2)), list(range(1, n-1, 2))]
        edges = []
        for group in groups:
            edges += [[u, v] for u, v in zip(group, group[1:])]+[[group[-1], n-1]]
        for v in groups[0]:
            work[v] *= 2
        work[-1] *= 3
    elif family == 'random_packing':
        es = {(u, v) for u in range(n) for v in range(u+1, n) if rng.random() < 2.5/n}
        for u in range(n-1):
            if not any(a == u for a, b in es):
                es.add((u, int(rng.integers(u+1, n))))
        edges = [list(e) for e in sorted(es)]
    else:
        raise ValueError(family)
    demands = [1]*n if resource == 'serial' else rng.integers(1, 5, n).tolist()
    capacities = [1, 1, 1] if resource == 'serial' else [3, 4, 4]
    speeds = [1, 2, 4]
    bandwidth = np.zeros((3, 3))
    for p, q in itertools.combinations(range(3), 2):
        bandwidth[p, q] = bandwidth[q, p] = int(rng.choice([2, 4, 8]))
    factors = rng.uniform(.5, 1.5, len(edges))
    if family == 'costly_downstream':
        for i, (u, v) in enumerate(edges):
            if v == n-1:
                factors[i] *= 3
    eligible = [[p for p in range(3) if demands[v] <= capacities[p]] for v in range(n)]
    average = float(np.mean([work[v]/speeds[p] for v in range(n) for p in eligible[v]]))
    unit_transfer = float(np.mean([f/bandwidth[p, q] for f in factors for p in range(3)
                                  for q in range(3) if p != q]))
    data = np.zeros((n, n))
    for (u, v), f in zip(edges, factors):
        data[u, v] = float(ccr*average/unit_transfer*f)
    inst = {'source': 'synthetic-large/'+family, 'family': family, 'seed': seed,
            'latent_id': hashlib.sha256(latent.encode()).hexdigest()[:20], 'n': n,
            'target_ccr': ccr, 'resource_mode': resource, 'speed_ratio': 4,
            'model': 'fixed', 'overlap': True, 'ticks_per_unit': 1,
            'edges': edges, 'workload': work, 'demands': demands, 'capacities': capacities,
            'speeds': speeds, 'bandwidth': bandwidth.tolist(), 'data_mb': data.tolist(),
            'latency': 0, 'cost_rates': [1, 4, 16], 'budget': None,
            'units': 'synthetic ticks and additive costs; not measured hardware'}
    return inst


def teacher(inst, decoder):
    started = time.perf_counter()
    seeds, _ = portfolio(inst)
    engine = decoder.engine
    candidates = [mapping_of(s) for s in seeds]
    values = [engine.evaluate(m, decoder.order) for m in candidates]
    best = min(range(len(values)), key=lambda i: values[i])
    mapping, value = candidates[best], values[best]
    rng = random.Random(inst['seed']+3917)
    feasible, attempted = 0, 0
    for _ in range(64):
        attempted += 1
        changed = list(mapping)
        for v in rng.sample(range(engine.n), rng.choice([1, 2])):
            changed[v] = rng.choice(engine.info['eligible'][v])
        if not engine.feasible(changed):
            continue
        feasible += 1
        candidate = engine.evaluate(changed, decoder.order)
        if candidate < value:
            mapping, value = changed, candidate
    solution = engine.evaluate(mapping, decoder.order, materialize=True)
    solution['algorithm'] = 'Offline-Portfolio-Capped-Mapping-Teacher'
    solution['search'] = {'attempted': attempted, 'feasible': feasible,
                          'simulations': engine.evaluations, 'converged': False}
    forced = decoder.rollout([[0., 0.]]*(engine.n*engine.k), 'initial',
                             forced_mapping=mapping, materialize=True)
    if forced['schedule']['tasks'] != solution['tasks']:
        raise AssertionError('teacher replay mismatch')
    return solution, forced['decisions'], time.perf_counter()-started


def specs():
    rng = random.Random(739153)
    for i in range(800):
        yield 'train'+str(i), rng.choice(FAMILIES), rng.choice([20, 40, 80]), rng.choice(['serial', 'pool']), 600000+i, rng.choice([0, .1, 1, 5]), rng.choice(['nonbinding', 'tight25', 'tight60'])
    for fold, count, offset in [('val', 1, 700000), ('test', 2, 800000)]:
        i = 0
        for family, n, resource, replicate in itertools.product(FAMILIES, [20, 40, 80], ['serial', 'pool'], range(count)):
            seed = offset+1000*FAMILIES.index(family)+100*[20, 40, 80].index(n)+10*(['serial', 'pool'].index(resource))+replicate
            for ccr, budget in itertools.product([0, 1, 5], ['nonbinding', 'tight25', 'tight60']):
                yield fold+str(i), family, n, resource, seed, ccr, budget
                i += 1


def build(out):
    records = {}
    journal = out/'dataset.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            records[r['key']] = r
    started = time.perf_counter()
    with journal.open('a') as f:
        for key, family, n, resource, seed, ccr, budget in specs():
            if key in records:
                continue
            inst = previous.g1.old.add_budget(base(family, n, resource, seed, ccr), budget)
            decoder = CachedBudgetDecoder(inst)
            heft = core.schedule(inst, 'HEFT')
            zero = decoder.rollout([[0., 0.]]*(n*3), 'initial', materialize=True)
            if zero['schedule']['tasks'] != heft['tasks']:
                raise AssertionError('zero-head decoder must exactly recover HEFT')
            r = {'key': key, 'instance': inst, 'reference': heft['makespan'],
                 'graph': previous.g1.old.graph_features(inst)[0]}
            # Measured independently of teacher/baseline generation.
            _, r['feature_seconds'] = previous.g1.old.graph_features(inst)
            if key.startswith('train'):
                r['teacher'], r['teacher_decisions'], r['teacher_seconds'] = teacher(inst, decoder)
            else:
                r['baselines'] = [core.schedule(inst, 'Greedy'), heft, core.schedule(inst, 'PEFT')]
                # Random chooses both a ready task and an eligible budget-safe device.
                r['random_schedules'] = [previous.g1.old.rollout(inst, [0.]*(n*3), 'joint', True, 131+seed*17+j)['schedule'] for j in range(5)]
            records[key] = r
            f.write(json.dumps(r, allow_nan=False)+'\n')
            f.flush()
            if len(records) % 100 == 0:
                print(f'large dataset {len(records)}/1448 {time.perf_counter()-started:.1f}s', file=sys.stderr, flush=True)
    latent = {fold: {r['instance']['latent_id'] for k, r in records.items() if k.startswith(fold)}
              for fold in ['train', 'val', 'test']}
    for a, b in itertools.combinations(latent, 2):
        if latent[a] & latent[b]:
            raise AssertionError('latent DAG leakage')
    with gzip.open(out/'dataset.jsonl.gz', 'wt') as f:
        for r in records.values():
            f.write(json.dumps(r, allow_nan=False)+'\n')
    return records, {k: len(v) for k, v in latent.items()}


def main():
    records, decoders = {}, {}
    for line in sys.stdin:
        q = json.loads(line)
        if q['command'] == 'init':
            records, counts = build(Path(q['out']))
            decoders = {k: CachedBudgetDecoder(r['instance']) for k, r in records.items()}
            result = {fold+'_ids': [k for k in records if k.startswith(fold)] for fold in ['train', 'val', 'test']}
            result['latent_counts'] = counts
        elif q['command'] == 'get':
            result = [{'id': k, 'graph': records[k]['graph'], 'reference': records[k]['reference'],
                       'feature_seconds': records[k]['feature_seconds'],
                       'teacher_decisions': records[k].get('teacher_decisions')} for k in q['ids']]
        elif q['command'] == 'rollout':
            result = []
            for item in q['items']:
                r = decoders[item['id']].rollout(item['logits'], q['mode'], q['sample'], item['seed'],
                                                materialize=q.get('include_schedule', False))
                r['id'] = item['id']
                # Cached physical compilation is charged at deployment too.
                r['decoder_preparation_seconds'] = decoders[item['id']].engine.compilation_seconds
                r['decoder_seconds'] += r['decoder_preparation_seconds']
                result.append(r)
        elif q['command'] == 'close':
            print(json.dumps({'closed': True}), flush=True)
            break
        else:
            raise ValueError(q['command'])
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
