# Published improvement magnitudes and actual teacher supervision

Accessed 2026-10-15. Provenance: ai-executed primary-source/code audit; interpretation and suggested thresholds are ai-suggested. Selected examples, not an exhaustive meta-analysis. No new training, neural calls, solver calls, or baseline reproduction.

## Metrics that must remain separate

Our I2-D4 table measures reducible current teacher makespan `(C_teacher-C_OPT)/C_teacher` on exact small instances, or valid intervals on unresolved larger instances. Published percentage improvements below compare algorithms `(C_baseline-C_method)/C_baseline`; they do not establish baseline-to-OPT room. Neither quantity alone guarantees a learned policy can attain the improvement.

The cited small table uses 64 paired physical bases with 7–10 tasks per budget/CCR cell. At CCR1/nonbinding the teacher has only 0.552% average reducible room in that exact slice; a 5% average reduction on those same cases is impossible. Tight25/CCR5 has 9.593%. These are not the main 20/40/80-task training/test distribution. I2-D4's larger known discovery slice has teacher lower endpoints 12.280%/10.891% at tight60/tight25, with only 12 base graphs, 14/36 source exact proofs, and no policy-reachability certificate.

Let C_fixed be the best schedule over all feasible device mappings under the fixed HEFT construction order and existing physical insertion scheme. Since the teacher is constructed by that same scheme, C_OPT <= C_fixed <= C_teacher. Its deployable neural scorer may impose further representational restrictions. Fixed construction order does not mean all tasks execute sequentially. The difference between C_fixed and C_OPT has not been quantified. One forward does not logically require fixed HEFT priority: a network could output priorities too, provided the method and comparisons retain the declared forward/decode budget.

## Primary-source examples

1. **WeCAN**, *A Learning Method with Gap-Aware Generation for Heterogeneous DAG Scheduling*, arXiv:2603.23249v2, 13 June 2026. This citation establishes a preprint, not an accepted-conference claim. Source: https://arxiv.org/html/2603.23249v2 . Table I: WeCAN-Greedy improves over best heuristic IPPTS by 8.47%, 10.05%, 8.48% on TPC-H-30/50/100. One network forward, one greedy schedule. Table III: WeCAN-S(256) improves over IPPTS by 6.22%/6.15% on RI-W-30/50, but selects among 256 schedule samples. Raw PPO-BiHyb makespans 24383/36958 and WeCAN 24099/35695 yield 1.1647459295%/3.4173927161% relative improvements (calculated from displayed means; not a mean of paired per-instance gains). Table II gives up to 13.46% vs best heuristic for sampled computation graphs, not the greedy value. Model and constraints differ from our hard-monetary-budget scenario.

2. **GiPH**, *Generalizable Placement Learning for Adaptive Heterogeneous Computing*, MLSys 2023. Official proceedings: https://proceedings.mlsys.org/paper_files/paper/2023/file/3e3eec95971350490e37a076fdc100ad-Paper-mlsys2023.pdf . Abstract/introduction report up to 30.5% lower completion times relative to search-based comparisons. Introduction, Sec.5 and conclusion explicitly describe makespan quality as comparable to HEFT. The 30.5% headline must not be stated as a HEFT gain or overall average. Iterative placement search, not one-forward/one-decode and not our hard-budget/cumulative-capacity model.

3. **AOM-SO**, *Adaptive-oriented mutation snake optimizer for scheduling budget-constrained workflows in heterogeneous cloud environments*, FGCS 175 (2026) 108118; online 2025. Author-hosted full paper: https://www.cs.newpaltz.edu/~lik/publications/Yanfen-Zhang-FGCS-2026 ; DOI https://doi.org/10.1016/j.future.2025.108118 . Abstract reports average 43.03% makespan reduction across its comparisons. The 41-task molecular dynamics case in Fig.10(b) discussion reports 7.07% average reduction over original SO (17.37% vs OSAM,14.39% vs MG-PRO,35.31% vs AEFT). Shows that comparator/workload matters. Population iterative search and serial VM model differ from our one-decode cumulative pools. Results not independently reproduced here. Existing full-text archive: refresh_20261012/aom2026.txt .

Additional checks not used for a new quantitative conclusion: MSLBL author PDF text contains Table7, but web screenshot failed cache retrieval. One-Shot original OpenReview PDF currently returns browser-verification page; do not infer original table values from a search snippet or later reproduction. BUDA's reported success-rate changes must not be mislabeled as makespan reductions.

## Actual teacher pipeline verified in our code

`experiments/large_dag_next/server.py:79`:

- Run ten heuristic variants on each training instance; extract mappings and replay them under the student's fixed HEFT construction order.
- Select lexicographically by `(makespan,cost)`; enforce budget feasibility rather than optimizing cost and makespan independently as equally weighted objectives.
- Make 64 seeded random proposals changing one/two task assignments. Discard infeasible proposals; keep lexicographic improvements. The cap is attempts, not successful moves or convergence.
- Force replay of the final mapping through the same decoder; store each task's selected eligible pool along with masks/features/offsets. These are hard action pseudo-labels, not OPT ground truth or teacher network logits.
- Original training archive: 800 configurations, 36,540 actions. Train/validation/test are separated by latent graph identity. Later diagnostic test teachers are not permitted for training.

`experiments/large_dag_next/train.py:65`: BC minimizes negative log probability of saved teacher actions; gradients train the student parameters. There is no separately trained teacher neural network, teacher parameter transfer, soft-logit distillation, or solver-based optimal labels in this recipe. Calling this a BC warm start is clearer than 'teacher-model parameter initialization'. Existing student pretrained weights can be loaded; they are not weights of an offline heuristic teacher.

`experiments/large_dag_next/train.py:129–130`: retained PPO includes a 0.02 teacher action likelihood anchor in addition to the clipped policy gradient and entropy terms. A potential restriction on surpassing the teacher is a hypothesis, not an established failure cause. Any anchor ablation must be controlled and report stable incremental makespan gains over BC.

## Interpretation

Several-percent gains over a strong comparator can be meaningful when replicated and accompanied by mechanism, generalization and runtime evidence; no universal acceptance threshold follows from these examples. A stable 3–5% advantage can be a useful engineering target, not an IWQoS rule or acceptance guarantee. Large gains vs weaker rules are not evidence of improvement over the teacher. If fixed-order mapping headroom is genuinely below 1–2% on the intended distribution, blindly adding features is unlikely to support a large quality claim; test the representation restriction and motivated problem distribution first. That 1–2% range is an AI-suggested triage guideline, not measured here.
