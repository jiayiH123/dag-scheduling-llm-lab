#!/usr/bin/env python3
"""Independent stdlib audit of serialized TRAIN schedules and conditional targets."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.structured_dataset import iter_rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); count = actions = 0; probability_error = 0.; offset_error = 0.
    for row in iter_rows(args.data, 'train'):
        x, s = row['input'], row['supervision']; n, k = x['n'], x['k']
        t, p, net = x['tasks'], x['pools'], x['network']; mapping = s['task_to_pool']
        edge_bytes = {(u, v): value for u, v, value in net['edges']}
        parents = [[] for _ in range(n)]
        for u, v, _ in net['edges']: parents[v].append(u)
        for u, v, data in net['edges']:
            comm = 0 if mapping[u] == mapping[v] else math.ceil(data/net['bandwidth_MBps'][mapping[u]][mapping[v]]*1000-1e-10)
            if s['starts'][v] < s['ends'][u]+comm: raise AssertionError('dependency/communication')
        events = [defaultdict(int) for _ in range(k)]
        for v, q in enumerate(mapping):
            if s['ends'][v]-s['starts'][v] != t['duration_ticks'][v][q]: raise AssertionError('execution time')
            events[q][s['starts'][v]] += t['demand'][v]; events[q][s['ends'][v]] -= t['demand'][v]
        for q in range(k):
            used = 0
            for time in sorted(events[q]):
                used += events[q][time]
                if not 0 <= used <= p['capacity'][q]: raise AssertionError('cumulative capacity')
            if used != 0: raise AssertionError('unclosed calendar')
        spent = 0; reserve = x['budget']['minimum']; ts = x['normalization']['time_ticks']
        completed = set()
        for v, d in zip(x['heft_order'], s['decisions']):
            minimum = min(t['cost'][v]); reserve -= minimum
            if not set(parents[v]) <= completed: raise AssertionError('construction order')
            denominator = max(1, max(c-minimum for c in t['cost'][v]))
            eft = d['candidate_end_ticks']; scores = [-value/max(1, ts) for value in eft]
            upper, lower = max(scores), min(scores)
            expected = [(f-upper)/max(1., upper-lower) for f in scores]
            for i, action in enumerate(d['available']):
                q = action % k
                start = d['candidate_start_ticks'][i]
                if eft[i]-start != t['duration_ticks'][v][q]: raise AssertionError('candidate duration')
                release = 0
                for u in parents[v]:
                    comm = 0 if mapping[u] == q else math.ceil(edge_bytes[u, v]/net['bandwidth_MBps'][mapping[u]][q]*1000-1e-10)
                    release = max(release, s['ends'][u]+comm)
                if start < release: raise AssertionError('candidate input release')
                if abs(d['budget_local'][i]-(t['cost'][v][q]-minimum)/denominator) > 1e-12:
                    raise AssertionError('local budget feature')
                offset_error = max(offset_error, abs(expected[i]-d['offsets'][i]))
            m = max(d['offsets']); logz = m+math.log(sum(math.exp(z-m) for z in d['offsets']))
            expected_logp = d['offsets'][d['selected']]-logz
            probability_error = max(probability_error, abs(expected_logp-d['selected_log_probability']))
            if d['candidate_start_ticks'][d['selected']] != s['starts'][v] or eft[d['selected']] != s['ends'][v]:
                raise AssertionError('selected calendar')
            if spent != d['spent_before'] or reserve != d['future_min_reserve']: raise AssertionError('budget history')
            spent += t['cost'][v][mapping[v]]; completed.add(v); actions += 1
        if spent != s['cost'] or spent > x['budget']['limit']: raise AssertionError('fee/limit')
        count += 1
    if max(offset_error, probability_error) > 1e-12: raise AssertionError('conditional numerical mismatch')
    result = dict(status='PASS', TRAIN_records=count, TRAIN_actions=actions,
                  max_offset_error=offset_error, max_logprob_error=probability_error,
                  independent_schedule_event_sweep=True, TEST_loaded=False,
                  scope='serialized TRAIN label physics and probability semantics; no neural optimization')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
