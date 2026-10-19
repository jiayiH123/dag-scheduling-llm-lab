# Locked protocol: structured scheduling simulation data v1

User accepted the immediately preceding proposal and explicitly requests execution. This protocol is committed before generating any new examples, engineering fixtures, labels or quality measurements. Data-only work; no Qwen model download, fine-tuning, new neural-policy score or final model TEST evaluation is part of this build. No scheduler/daemon is available; foreground work and resumable artifacts provide continuity.

## Counts and split contract

Config: configs/structured_scheduling_v1.json. Main independent base DAGs: TRAIN10000, VAL500, TEST1000. Two budget configurations share each base's topology, workload, platform and network: main20000/1000/2000 records. Additional OOD-size400bases at160/200tasks and OOD-pools400bases at8pools, each two configurations,800records each. OOD changes task count or pool count separately. First1000TRAINbases/2000records form the pilot, then are retained as a subset of main TRAIN, not counted twice.

All-source graphs are newly simulated, with five graph motifs and O(N) edges bounded by4N. N20/40/80/120 and motif counts balanced; each physical factor separately shuffled in balanced cycles. Directed WL16 topology hashes group/deduplicate across all roles before platform/budget expansion. Only topology duplicates or malformed inputs can be rejected. No graph is selected/discarded by teacher, baseline or student performance. Source seeds, duplicate attempts and actual counts are recorded. All variants of a base remain together.

TRAIN receives labels; VAL receives HEFT reference only for the current trainer bridge, no training targets. TEST/OOD receive problem inputs and static feasibility/serialization checks only; no baseline/teacher/learned scheduling performance measurement. No public-source TEST data or labels are used. Existing WfInstances/TPC-H remain separate application data; none is renamed as agent trace.

## Physics

Same original fixed-overlap model: nonpreemptive tasks, single cumulative capacity, no link contention, communication may overlap execution, same-pool transfer0, latency0. Relative speeds contain1 andH withH2/4/8 and random intermediate speeds; positive integer fee rates are round(10*speed^gamma*jitter), gamma uniform[1.2,2], jitter[.9,1.1]. Pool order shuffled. Capacities4/8/12 and demands1..4; light/mixed/heavy demand distributions. Workload is lognormal clipped to[.02,10] reference seconds, duration ceil(workload/speed*1000), fee duration*rate. These are simulated units, not provider tariffs or hardware measurements.

Symmetric links sampled from10/100/1000MBps with[.8,1.2]jitter. Edge-size lognormal factors clipped[.25,4], then scaled so mean cross-pool communication seconds / mean task/pool execution seconds targets CCR0/.1/1/5; actual quantized CCR saved. Bandwidth/payload and rounding assumptions are explicit.

Bmin=sum per-task minimum eligible fee. Bfast=fee of per-task fastest mapping, tie by fee/pool. Paired eta(.1,.5),(.25,.75),(1,nonbinding); finite nonbinding B=sum per-task max eligible fee. No claim eta1 is universally nonbinding with perturbed prices. Every input has an analytic serial feasible witness: choose each task's cheapest pool, serialize topologically with any required communication; each fits capacity and total feeBmin<=B. Integer budget and rate rules follow frozen core.

## Labels and representation

Unchanged ten-rule portfolio from vendor/search_controls.py, mapping re-evaluation under fixed HEFT order,64local one/two-task proposals as in daglab.reference.teacher. Select lexicographic(makespan,cost). Retain source-rule values, remap work and winning mapping; no solver optimality or neural teacher claim. Small solver subset is optional diagnostic, not needed for build acceptance and not silently substituted for fixed-order reachable labels.

Replay TRAIN mapping through NormalizedEFTDecoder LOCAL; store eligible/affordable pools, exact candidate EFT, normalized offsets, local budget feature, spent/reserve history, choices and teacher start/end arrays. Validate physical schedule and replay equality. Scores usea+w*phi+relativeEFT in[-1,0]; teacher map rather than invented a/w regression is the target. A bounded artificial +/-7 assignment head must recover the teacher mapping as an engineering expressibility check, not a learned result. Teacher prefix history is offline BC only; PPO requires its own sampled-prefix replay.

Master format separates input from TRAIN supervision and provenance/quality. Inputs contain sparse edges, all duration/fee arrays, platform/network/budget and input-derived normalizers. Numeric adapter builds complete global/task/pool/edge/link/assignment prefix followed by all known pair-query tokens; no label/candidate-score input. All query positions must see the entire instance, including all pair duration/fee records. Max numeric positions4096, no graph truncation. Raw JSON tokenizer length is not thereby verified. Qwen/GNN/Attention share the same physical data and decoder; larger-backbone benefit remains untested.

## Checks, pilot and publication

Gates: source determinism, acyclicity, physical values, static budget/capacity feasibility, no topology overlap, TRAIN-only targets, forced replay and schedule validation, numeric query ordering/length, label likelihood consistency, and original frozen integrity. Pilot reports full TRAIN teacher/HEFT/MSLBL ratios and subgroups descriptively, with no performance-based filtering or promotion. If an implementation gate fails, fix/code-commit and rerun affected engineering checks before continuing; record attempts.

Resumable bounded gzip JSONL shards, manifests, source/config hashes and standard-library verification; no Git LFS or external first-run data fetch. Ship actual data and model-independent readers/numeric adapters plus old-trainer conversion commands. Keep Git blobs below50MiB; no weights/credentials. Push additive portable repo changes for pull-only company access, verify remote ref. New ready data does not mean Qwen policy/training is implemented. ARA epilogue only after main execution/handoff completes; existing default and outer37 learning loop unchanged.
