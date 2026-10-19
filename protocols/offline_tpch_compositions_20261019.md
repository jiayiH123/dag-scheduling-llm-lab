# TPC-H offline composition supplement, before generation

The author checkout has154query DAGs (22query IDs x7input scales), each2-18stages,
plus618rawdata/visualizationfiles. Individual query graphs are retained in full.
They are too small alone to represent the author's50/100/150-query workloads.

Also generate1000multi-query DAGs:800TRAIN/100VAL/100TEST, query counts5/10/20
cycled deterministically per split. Shuffle query IDs1..22 with seed20261019;
first14IDs TRAIN,next4VAL,last4TEST. All7input scales available within each role.
Each composition is a disjoint union of full query DAGs, with no invented
cross-query dependencies. Preserve component IDs/scales and source metadata.
Reject duplicate composition IDs and topology fingerprints crossing splits;
within-split repeated topologies are counted and disclosed. These are generated
compositions of measured query records, not1000independent real traces, original
author train/test reproduction, or a certified official TPC-H database result.

Use author preprocessing of first-wave/fresh-duration samples and mean duration
in source sample units, interpreted as milliseconds for communication modeling.
Stage demand is first-wave+rest-wave task count at the author's first available
executor setting. Cumulative pool capacity600, relative speeds1/2/4 repeated for
3/6pools, rates1/4/16. Missing measured edge traffic is simulated, with stable
per-edge factors .5..1.5 and a reference mean delay near mean fastest stage time
on the slow network;100/400MBps profile is10xfaster than10/40MBps profile.
Budget etas.25/.6, fixed overlapping transfers, no contention. Eightvariants per
base, all grouped by source role. Raw individual queries are separately marked
primitive-only; their common shapes must not be presented as a strict structural
holdout across query templates. New composite protocol is not model-selected.

No teacher labels generated for VAL/TEST. All data must be usable from ordinary
Git without external downloads; adaptation schema, generated provenance and
resource/network/time assumptions remain explicit. No modeltraining or baseline
win is inferred from this engineering handoff.
