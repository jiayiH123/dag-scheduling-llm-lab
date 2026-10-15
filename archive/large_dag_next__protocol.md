# G1-P4: larger DAGs and matched budget-score controls

Status: FROZEN before execution. Exploratory development pilot with prespecified gates, not a final paper benchmark. Resume after transport disconnection; existing checkpoints preserved.

Question: do gains persist at 20/40/80 tasks; does PPO improve shared BC; does the explicit global budget denominator outperform a task-local cost denominator? Keep GLOBAL makespan, fixed overlapping lags, nonpreemptive cumulative resources and hard additive costs. Exactly one whole-graph neural forward, deterministic analytic decoding, no search/beam/retries/best-of-K at inference.

Borrowed components: GiPH task-device product graph, BC plus PPO and HEFT rank/EFT. Primary-source cards and snapshots already in literature/. No claim to invent these components or prove latest-paper novelty.

Data: devices speeds[1,2,4], rates[1,4,16]. Serial capacities[1,1,1]/demands1; pooled capacities[3,4,4]/demands1..4. Families width4 pipelines, broad fork-joins, asymmetric chains with costly downstream, sparse packing probability2.5/n plus terminal paths. Workloads multiples4; random actual bandwidth and edge factors. CCR0/1/5 (train also.1), budgets eta1/.25/.6 between actual Bmin/Bfast. No contention, arrivals, DVFS, replication or added dimensions.

800 training latent graphs (one random variant each), seed600000+i. Validation24 new latent graphs x9 CCR/budget variants =216 configurations, seed700000+. Test48 new latent graphs x9 =432 configurations, seed800000+. Validation/test factorial family4 x size3 x resource2 x replicate(1/2) x CCR3 x budget3. Assert disjoint latent IDs across splits;432 configurations are48 independent graphs. Primary subset CCR1 and tight25/60 excluding budget-degenerate (expected96 configurations).

Baselines: common-model Greedy, budget-adapted HEFT, PEFT, Random mean of5 feasible joint ready-task/device rollouts. Same min-cost reserve mask and physics. Random5 is expectation, not best-of5. No CP labels or OPT-gap claims.

For seed11/22/33 initialize from its validation-selected G1-P1 BC. Budget head starts zero and is frozen during shared large-DAG BC:200 updates x batch8, Adam.001, gradclip1, mean teacher categorical CE. Select at0/every50 by mean validation C/HEFT. Both PPO arms start from IDENTICAL full weights/policies.

Offline training-only teachers: ten fixed rules (HEFT/PEFT/JoinLookahead/CostFirst/MSLBL plus five prices), replay mappings in fixed HEFT order;64 seeded one/two-task proposals retain lex(C,cost) improvements. Save all attempted/feasible counts and time. Completeness=false, not optimum or neighborhood closure. Distillation alone does not establish novel RL contribution.

Both arms BudgetFeedbackActor14946 parameters, same12 features/two message-passing layers/two active heads a,w. One forward emits all scores. Same hard mask, fixed HEFT order, prepared calendars. Score=a-EFT/mean_fastest_duration+w*phi.
INITIAL: phi=(cost-min_task_cost)/max(1,B-sum_all_min_cost).
LOCAL: phi=(cost-min_task_cost)/max(1,max_eligible_cost-min_task_cost).
Both heads actively affect rankings. LOCAL still sees budget eta in input and uses hard budget mask: this isolates only explicit score normalization, NOT all budget information, and not identical function classes. No unused-head parameter-match claim.

Each arm actor-only PPO300 updates max x batch8,4 reuse epochs, clip.2, KL stop.03. Terminal advantage=(greedy_C-sampled_C)/HEFT_C, detached counterfactual; mean teacher CE anchor.02, entropy.001→0, gradclip1; Adam3e-4 warmup20/cosine3e-5. Same180s wall ceiling per arm, excluding initial validation and including periodic validation. Record actual steps/time if cap binds; don't then claim equal updates. Same training/RNG sequence. Only validation selects checkpoint0/every50/final cap. Check finite losses/gradients. Vectorized conditional calculations preserve per-decision clipped PPO objective.

After ALL selection, evaluate BC, both PPO and frozen previous micro PPO for3 seeds on SAME fresh test. Choose one deployment candidate by lowest validation among6 PPO candidates (step0 BC fallback included), never test-best seed/arm. Save negative branches.

Prespecified scale gate: primary mean paired HEFT gain >=5% and positive overall paired gains vs each Greedy/HEFT/Random in >=2/3 seeds. RL contribution gate: overall relative Cmax improvement >=1% vs same-seed shared BC in >=2/3 seeds. INITIAL normalization gate: primary relative gain >=2% vs LOCAL in >=2/3 seeds. Gates separate; baseline win alone does not establish RL or normalization benefit;step0 is BC fallback.

Report all seeds, size/budget/CCR/family/resource strata; mean paired percent, ratio of total makespans, win/tie/loss and worst degradation. Bootstrap2000 paired resamples of48 latent clusters preserving9 variants per graph (16 clusters per size). Training-seed uncertainty remains separate, n=3. Real-trace confirmation/calibrated platform still outstanding.

Critical checks: prepared vs old INITIAL decoder on64 micro plus24 larger schedules, exact task/communication/order records and decision/features/probability agreement;zero head recovers HEFT. Every built config zero schedule compared with full HEFT. Independently validate all saved teachers, baselines, random, neural schedules with unchanged event sweep/cost ledger. Exactly1 full forward on neural evaluations.

Timing: charge features, physical preparation, neural forward, deterministic decoder and saved-output checker in CPU component sum;RPC separate, not deployment end-to-end. Training rollouts omit materialization/checker; saved outputs independently validated. Shared prepared implementation is engineering, not neural speedup evidence.

Recoverability: append dataset journal per completed record, save selected model/curves after every completed stage. Resume skips finished stages and refuses changed hashes/finished-run overwrite. An incomplete arm restarts from shared BC; finished stages preserved. No persistent scheduler/background agent.

If failed scale: inspect teacher gaps, size/budget failures and training before next controlled change. Repeated stagnation invokes creative-thinking-for-research plus brainstorming-research-ideas. ARA manager only at session epilogue.
