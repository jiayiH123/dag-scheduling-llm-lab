# Single-Pass Learned Policies for Budgeted Communication-Aware DAG Scheduling

**Working draft, 2026-10-19.** This is an implemented-method and evaluation draft,
not a submission-ready claim of superiority. The small public pilot has completed;
reserved TEST results and original-paper reproductions are not available. All
modules are attributed below. Final contribution wording depends on measured
benefits over strong adapted controls.

## Abstract

Scheduling dependent tasks across heterogeneous resource pools requires balancing
execution speed, communication delay, concurrent capacity, and a finite execution
budget. A placement that finishes the current task early can consume capacity or
budget needed by later tasks, while evaluating many complete schedules introduces
deployment overhead. We investigate a policy that encodes the complete workflow
once and delegates state-dependent feasibility and timing to an exact constructive
decoder. The policy combines task–pool representations with explicit communication
messages and either normalized graph aggregation or factorized attention. It is
initialized with offline heuristic schedules and improved using complete-schedule
policy-gradient or preference objectives. A training-only divergence penalty
controls deviations from the supervised policy without adding a deployment model.
Our prototype preserves dependency, capacity, and budget feasibility with one
policy forward and one final construction. Preliminary synthetic development
experiments show small and variable optimization gains and a tradeoff between
average completion time and tail regressions; they do not establish superiority
over strong budget-aware heuristics. A public-workflow pilot also identifies a
score-range mismatch that prevents some demonstration actions from being chosen
by any bounded policy output. Normalizing the timing prior improves the selected
GNN validation result, but it remains worse than the adapted budget-aware control.

## 1. Introduction

Distributed scientific workflows expose parallelism through directed acyclic
graphs (DAGs), but exploiting this parallelism requires choosing where each task
runs. Faster pools can reduce execution time while increasing monetary cost.
Placing a child away from its parents introduces data transfer delay, and multiple
otherwise ready tasks may compete for the same pool capacity. These decisions
are coupled through the final workflow completion time.

Classical list schedulers make fast decisions using task priorities and estimated
finish times. HEFT and PEFT provide important reference algorithms for
heterogeneous execution and communication [1,2]. Budget-aware controls can repair
many apparent weaknesses of simpler rules, so an improvement over HEFT alone does
not establish an advantage over relevant methods. Learning-based placement and
one-shot priority methods provide additional precedents [3,4]. Our target is the
specific combination of cumulative pool capacity, additive hard budget, fixed
communication delays, and a single deployed policy evaluation.

The working design separates two responsibilities. A neural representation
captures task, pool, and communication relationships once. A constructive decoder
then computes the actual timing and feasible choices as earlier assignments
change the calendar and remaining budget. Training can evaluate many candidate
schedules, but deployment retains one deterministic construction. This avoids
equating one encoder call followed by many sampled schedules with one final
scheduling decision.

The current prototype evaluates three questions: whether richer physical inputs
and normalization improve supervised compression; whether factorized attention
adds value beyond a normalized GNN; and whether optimization after imitation can
improve deterministic makespan without severe regressions. These are research
questions, not established new algorithmic principles or a claim that no prior
work considers budgets or communication.

## 2. Model and objective

Let G=(V,E) be a DAG and P the available pools. Task v has demand q_v and execution
duration d_vp on pool p. Pool p has cumulative capacity R_p and cost rate r_p.
Its assignment cost is c_vp=d_vp r_p. For dependent tasks (u,v), the fixed delay
delta_uv(p,q) depends on their assigned pools and input data. Same-pool delay is
zero; transfers may overlap and do not contend for links in this model.

The scheduler chooses assignments m_v and start times s_v to minimize

`Cmax = max_v(s_v + d_v,m_v)`

subject to precedence plus communication,

`s_v >= s_u + d_u,m_u + delta_uv(m_u,m_v)` for every edge (u,v),

the cumulative capacity condition

`sum_{v: m_v=p, s_v<=t<s_v+d_vp} q_v <= R_p`,

and the hard execution budget `sum_v c_v,m_v <= B`. Tasks are nonpreemptive.
Our present decoder fixes HEFT's construction priority, which does not fix the
physical execution order: calendar insertion permits overlap and earlier slots.
This decoder does not express every schedule available to a global optimizer.

## 3. Design

### 3.1 Complete-instance task–pool representation

Each potential task–pool assignment is a product node. The basic input includes
task workload and demand, predecessor/successor counts, candidate duration and
cost, pool speed/capacity/rate, a critical-path rank, and the initial budget level.
The extended input appends optimistic successor cost, depth, relative candidate
duration, and incoming/outgoing communication statistics. Product edges preserve
the normalized communication delay between feasible assignment pairs.

This representation follows the general product-graph placement idea of GiPH
[3]. The optimistic cost table borrows the existing PEFT recurrence [2] and is
an explicit prior rather than a learned discovery. We separately evaluate
information additions and an analytic bounded-OCT scoring control.

### 3.2 Normalized graph aggregation and factorized attention

The normalized GNN aggregates forward and backward messages along DAG edges.
Delay embeddings enter messages explicitly instead of solely suppressing them
with inverse-delay weights. Layer normalization stabilizes the resulting node
representations. A global eligible-node average supplies additional context.

The attention variant retains directed messages, then performs attention across
tasks for each pool and across pools for each task. Directional DAG and pool-delay
biases preserve scheduling relationships. This is an adaptation of scheduling
attention ideas such as ReSched [5], not a reproduction of its sequential FJSP
policy. It avoids a dense attention matrix over all N*P product nodes, but still
has quadratic task attention and communication-edge preprocessing costs.

Both variants emit two parallel scalars a_vp,w_vp bounded by8*tanh(raw/8).
The deployed score is

`score(v,p) = a_vp + w_vp * phi(v,p) + o(v,p)`,

where phi is the local normalized upgrade cost and EFT is recomputed from the
current assignment history and resource calendars. The network is not invoked
again during construction. The original prior is `o=-EFT/scale`. On the public
training traces, it can dominate the bounded neural output. For teacher choice j
and competitor i, the largest possible margin under independently bounded heads
is `16 + 8*(phi_j+phi_i) + o_j-o_i`. A negative bound means that action cannot be
selected at that teacher prefix, regardless of network capacity. A TRAIN-only
audit finds411 such decisions among4992, affecting19of64training configurations.
This is a limitation of our score interface, previously also investigated on
synthetic data, rather than a newly discovered limitation of DAG scheduling.

The calibrated variant uses `o'=(o-max(o))/max(1,max(o)-min(o))` over currently
feasible pools. It preserves zero-output EFT ordering and bounds the prior in
[-1,0]. Physics, budget masks, construction priority and cap8 stay fixed. This
removes the particular bound obstruction but does not prove that a shared neural
network can represent every desired mapping. Forced replay of all64training
teachers preserves their exact schedules;96zero-output checks match HEFT.
These coefficients introduce neither learned task ordering nor a skip action.

### 3.3 Exact constructive feasibility

For each task in the construction list, the decoder excludes ineligible pools
and assignments that would leave insufficient money for the minimum-cost
assignment of all unscheduled tasks. It computes parent release times including
communication, finds the earliest cumulative-capacity slot on each surviving
pool, and selects the highest score with deterministic tie-breaking. The selected
task is inserted into the calendar and its cost is charged.

Reserving minimum future cost preserves at least one budget-feasible continuation
under this model when B initially exceeds the sum of minimum task costs. The
reserve does not optimize the value of future upgrades. Every materialized
schedule is independently checked against the authoritative physical evaluator.

## 4. Training

### 4.1 Offline demonstrations

The teacher is a program, not a neural model. It runs the existing ten-rule
mapping portfolio, replays assignments through the student's construction order,
then makes64 seeded local assignment proposals and keeps feasible improvements.
This is not an optimality certificate or a converged local search. Training-only
action pseudo-labels initialize the policy with cross-entropy. Each architecture
has its own BC checkpoint; subsequent objectives branch from exactly that state.

### 4.2 Complete feasible candidates

For each training instance, one network output produces B complete candidates:
one deterministic schedule and B-1 stochastic schedules. Each trajectory stores
its own available pools, selected actions, budget features, and EFT offsets. A
candidate's probability is recomputed under that candidate's prefix history.
Using another trajectory's masks or calendars would optimize the wrong policy.

The actor-only PPO control uses the deterministic schedule as its return baseline
and clipped likelihood ratios for stochastic candidates [6]. It has no critic or
GAE. Self-imitation trains on the best sampled schedule only when it improves on
the deterministic one. The ordinary preference control compares the best
candidate with retained inferior candidates. The BOPO adaptation [7] scales its
logistic loss by their objective ratio, using mean selected-action log likelihoods.
Equal candidate counts do not imply equal optimizer work; both are recorded.

### 4.3 Training-only control of policy drift

Hard feasibility prevents illegal schedules but does not prevent legal schedules
with poor completion times. We test adding a mean conditional divergence

`lambda * KL(pi_theta(.|candidate_prefix) || pi_BC(.|candidate_prefix))`

to PPO, while retaining the same teacher-action anchor used by the control.
The BC policy is frozen and evaluated only during training. No second model,
baseline fallback, candidate search, or trajectory comparison is deployed.
This is standard regularization; its value must be justified by measured quality
and tail behavior rather than presented as a new general RL algorithm.

## 5. Evaluation draft

### 5.1 Questions and controls

Primary quality is absolute global makespan, accompanied by mean paired gains
against each baseline. We report means normalized by HEFT for checkpoint
selection, plus wins/losses, worst regressions and graph-grouped intervals.
Ratios normalized by HEFT and paired percentages normalized by BC are different
averages and must not be substituted for one another.

Controls include Greedy, RandomMean5, HEFT, PEFT, adapted MSLBL, the ten-rule
portfolio and Teacher64 as a quality/time reference. Borrowed formulas and
model adaptations are distinguished from author-code reproductions. Test labels
are not used for training or model selection. A single deployment uses one full
forward and one deterministic decode; best-of-B is a training diagnostic only.

### 5.2 Existing synthetic development evidence

These measurements use the original800 synthetic training configurations and
216knownvalidation configurations from24base graphs. They are not final TEST
results. The selected architecture's reported checkpoints use the same knownVAL.

| Policy | MeanC/HEFT across3seeds | Mean paired gain over ownBC |
|---|---:|---:|
| Attention BC | 0.85918 | reference |
| Attention PPO | 0.84424 | +0.8745% |
| Attention BOPO-adapted | 0.85048 | +0.6977% |
| Attention PPO+BC-KL1 | 0.84827 | +0.9064% |

The three ordinary PPO paired intervals individually include zero. Mild BC-KL
reduces p95BC-relative regressions in all three seeds but slightly worsens mean
C/HEFT compared with ordinary PPO. Stronger KL further reduces exploration and
average improvement. The current neural results do not surpass MSLBL/portfolio/
Teacher64 on these development data. These observations motivate a stability
study, not a completed overall-quality claim.

### 5.3 Small public-workflow development experiment

The frozen WfInstances selection [8] contains8TRAIN and4VAL execution traces from
Montage,1000Genome,Epigenomics,SRA Search,SoyKB andSeismology. Observed graphs
contain22–233tasks. Measured graph/runtime/filebytes are preserved;3or6pools,
unit slot demands, capacity2, relative speeds/prices, network profiles and budgets
are declared simulation parameters. Eight configurations per trace yield64TRAIN
and32VAL records. Three additional TEST paths remain reserved and unfetched.

The pilot compares normalized GNN and Attention with BC/PPO/BOPO-adapted/
PPO+BC-KL1 at seed11, using200BC updates and20optimization updates per arm.
Both the original and calibrated score interfaces use the same data, seeds,
training budgets and physical controls. All224baseline objectives match exactly
across these two phases. Reported checkpoints are selected on the32VAL records.

| Policy | Original prior C/HEFT | Calibrated prior C/HEFT |
|---|---:|---:|
| GNN+LayerNorm BC | 0.91367 | 0.75993 |
| GNN+LayerNorm PPO | 0.91176 | 0.75993 (step0) |
| GNN+LayerNorm BOPO | 0.91120 | 0.75993 (step0) |
| GNN+LayerNorm PPO+BC-KL1 | 0.91272 | 0.75993 (step0) |
| Attention BC | 0.92129 | 0.87521 |
| Attention PPO | 0.92129 (step0) | 0.76617 |
| Attention BOPO | 0.91830 | 0.77618 |
| Attention PPO+BC-KL1 | 0.91578 | 0.76492 |
| Adapted MSLBL / Portfolio | 0.65456 | 0.65456 |
| Teacher64 | 0.65112 | 0.65112 |

Calibrated Attention PPO improves its ownBC by4.83% in the mean paired metric,
with a graph-grouped exploratory interval[-5.97%,16.18%]. The calibrated GNN's
strongest result is BC: subsequent optimization selects step0, and final PPO
instead regresses by9.62% versus BC in the paired metric. Reporting step0 as an
RL gain would be incorrect. The selected GNN is24.01% shorter than HEFT on
average, but13.88% longer relative to adapted MSLBL. The latter comparison must
remain visible alongside the weaker-baseline improvement.

Eight TRAIN workflows mean eight original DAGs, expanded into64platform/budget
configurations. Four VAL workflows similarly produce32records; these are not
32independent workflows. This small trace/scale split does not establish
application-family generalization. Subsequent main evidence requires more
independent public graphs and a declared independent confirmation run.

### 5.4 Remaining ablations and limitations

The next minimal ablations are capacity-matched normalized GNN versus Attention,
removal of explicit communication messages, removal of the OCT feature, and
ordinary PPO versus the same training budget with mild BC-KL. No broader full
Cartesian module sweep is planned until these answer a concrete question.

Current limitations include a fixed construction order, bounded score range,
simulated hardware/costs, no network contention, scalar pool capacity, limited
public development sample size, and no deployed execution validation. A cold
CPU timing check includes feature preparation, output transfer, JSON RPC and
construction; it is not evidence of optimized GPU serving throughput.

## 6. Related work draft

**Heuristic scheduling.** HEFT/PEFT [1,2] guide fast heterogeneous assignments.
Budget-aware methods must remain quality controls, not be omitted in favor of
weaker rules. Our exact cumulative-pool adaptation changes resource semantics
from serial-machine formulations.

**Learned graph scheduling.** GiPH [3] performs iterative placement improvements.
DAG-CAI-DR [4] learns one-shot priorities followed by heuristic placement. Our
current prototype instead fixes construction priority and learns assignment
scores; neither formulation implies a first-ever single-shot scheduler.

**Scheduling representations and optimization.** ReSched [5] supplies relevant
attention ideas for a different, sequential scheduling problem. BOPO [7] supplies
the preference objective. Combining these ideas is an adaptation whose benefit
requires controlled evidence. Ordinary clipped policy optimization [6] remains
an essential simpler comparator.

## 7. Current conclusion

The prototype implements a strict single-forward policy with an exact hard-
constraint decoder and reproducible training controls. Existing development
results support further investigation of representation normalization, score
calibration and training stability. The public pilot demonstrates an interface
failure and a useful repair in this implementation, but does not establish
superiority to strong adapted controls or sufficient novelty for publication.

## Verified primary-source pointers

1. HEFT/CPOP,TPDS2002: https://doi.org/10.1109/71.993206
2. PEFT,TPDS2014: https://ieeexplore.ieee.org/document/6471969/
3. GiPH,MLSys2023: https://proceedings.mlsys.org/paper_files/paper/2023/file/3e3eec95971350490e37a076fdc100ad-Paper-mlsys2023.pdf
4. DAG-CAI-DR,NeurIPS2025: https://papers.nips.cc/paper_files/paper/2025/file/dc16950e510bcd573783f5c2d7d8cc31-Paper-Conference.pdf
5. ReSched,ICLR2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html
6. PPO,authorpaper2017: https://arxiv.org/abs/1707.06347
7. BOPO,ICML2025: https://proceedings.mlr.press/v267/liao25a.html
8. WfInstances/WfCommons: https://github.com/wfcommons/WfInstances ; framework DOI10.1016/j.future.2021.09.043

These are source pointers checked in the project, not a finalized bibliography.
IWQoS year's CFP, formatting and submission policies are not fixed by this draft.
