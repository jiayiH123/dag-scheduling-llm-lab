# Module screen, locked before new local results

Classification: EXPLORATORY development screen, not an unseen public-data claim.
User-provided remote pilot report is not independently reproduced: seed11,
GNN warmstart versus randomly initialized Attention, BC50/RL20, weak paired
increments and intervals crossing zero. Its code/results are not available locally.

Hypotheses: richer complete-instance physics improves supervised compression;
explicit directed communication messages or factorized task/pool attention help
over an information-matched model; complete-solution optimization improves the
same model's deterministic deployment over its own frozen BC.

Data: bundled original synthetic TRAIN/VAL only (800/216 configurations). No TEST
evaluation or TEST labels. Local cheap screen uses at most128 stratified TRAIN
configurations,72 VAL configurations; all selected keys/hashes are saved. The
full216 known VAL is used once per saved candidate for final development reporting.
Splits remain grouped by latent graph. This is not the new public main experiment.

Architectures: legacy12 GNN, physical GNN, physical GNN with explicit-delay edge
messages, information-matched axial task/pool attention, physical MLP, and explicit
GNN with bounded static OCT score prior. Hidden32/layers2; no historical weights.
All prediction heads start at zero. Report parameter counts; this first screen is
not parameter matched. The physical tensor representation is shared. OCT is the
existing adapted PEFT recurrence; an analytic OCT-only score is a separate control.
The initial head zero reproduces HEFT except the explicit OCT-prior arm, whose
initialization is compared to its own OCT-only control. Fixed HEFT construction
order, LOCAL reserve mask, capacity/communication/budget checker and cap8 unchanged.

Local budget: seed11, BC80 updates, batch4, LR1e-3; evaluate every20. Rank six BC
models by the fixed VAL meanC/HEFT, then train the top2 using four objectives:
PPO, best-candidate self-imitation, unweighted best-anchored logistic pairs, and
objective-ratio BOPO-adapted. Each arm starts from the same architecture's BC.
RL20 updates, batch4, B8 actual complete candidates including1greedy+7random, K4
retained candidates; LR3e-4, teacher-anchor0.02, PPO4epochs versus1preference epoch.
Report candidate/gradient/forward counts and wall time rather than claiming equal
optimizer work. No early stopping that silently gives one arm additional updates.

Checks before trusting metrics: constraint-valid saved schedules; exact own-prefix
log-probability reconstruction to numerical tolerance; independent old/current
policy likelihoods; one complete model forward and one deterministic decode per
deployment; a TRAIN one-batch-overfit diagnostic; train/eval dropout0; finite
losses/gradients. Device-aware code supports CPU/CUDA without changing vendor.

Selection: step0 BC is eligible for every optimizer, and all last checkpoints are
saved as well. Select by meanC/HEFT; also report mean pairedpercent gains over ownBC
with latent-graph clustered exploratory intervals, wins/losses and worstregression.
Do not infer positive RL if only the checkpoint-selected point or best-of-B wins.
Final knownVAL controls: Greedy,HEFT,PEFT,MSLBL,ten-ruleportfolio,Teacher64.
No claim about stable seed-level gains until at least3seeds and new public folds.

Prior attribution: GiPH product-graph communication; ReSched scheduling attention;
PEFT optimistic cost table; BOPO best-anchored objective-guided preference training.
These are adaptations, not original-paper reproductions or new-principle claims.
CAI priority comparability does not change fixed-order assignment probabilities;
WeCAN skip/learned order is deferred to a separately labeled action-space study.

Continuation: server may raise BC/RL caps, B and seeds in a new locked config after
local screening. Do not overwrite old runs or server-only files. New code is under
daglab/module_lab and scripts/run_module_lab.py, avoiding remote pilot filenames.
No scheduler/cron is available locally; this is a bounded foreground cycle with
saved checkpoints, not a claim of a continuously running remote agent.
