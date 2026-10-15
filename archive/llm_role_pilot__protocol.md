# LLM role pilot — DRAFT, NOT EXECUTED

2026-10-15;ai-suggested design in response to user's role/input question. No new model, server connection, generation, training or neural evaluation. This is not a locked runnable protocol: exact checkpoint, server, training/data budgets, score gates and runtime target remain to be specified and committed before execution.

## Question

Can a pretrained language-model backbone improve the existing numerical task-pool policy beyond BC and adapted strong baselines, while preserving one full backbone call and one physical decode? Distinguish that from an offline LLM designing better heuristic code.

## Preconditions

- Quantify fixed-order feasible mapping room and student-to-teacher loss; current globalOPT certificates alone do not diagnose either.
- Check current GNN learning curves at longer budgets before treating parameter count as the cause of weak PPO.
- Retain exact physics, graph-identity splits, baseline implementations and archived teacher recipe. Never train from retrospective TEST labels.
- Pin actual server/checkpoint/tokenizer and document license/resources before a remote run. None is assumed accessible in this draft.

## Stage A: no-deployment-LLM role

Use ReEvo/EoH-style offline proposals for one pure scoring function using critical-path relevance, normalized compute/upgrade cost, successor communication and remaining budget. The runtime interface exposes numerical features; evaluator owns feasibility, resource insertion and makespan. Freeze the code after training/selection data. Compare new rule to old individual rules, portfolio and frozen teacher; include generated heuristic as its own baseline. No claim that this is a learned neural scheduler or a guaranteed new idea. LLM calls and all simulation costs belong in offline design accounting.

## Stage B: isolate representation

GNN, small non-pretrained Transformer control, and small server-provided pretrained backbone with parallel policy heads. Backbone starts frozen; optionally add matched adapters later. Same information, a/w scoring range, fixedHEFTorder, BC targets, reserve mask and decoder. Serialize full sparse instance prefix plus all task-pool queries, then one forward and shared heads. No text generation or reflection loop. The small Transformer is a practical baseline, not an equal-total-parameter isolation; if LLM wins, add a same-architecture/random-initialization control with matched trainable modules to test pretrained-weight effects.

For each candidate report BC and matched BC+PPO,>=3seeds, identical graph splits, validation-only checkpoint selection and a fresh test fold generated after settings are fixed. Lock new instance counts/training updates before running; initial800train labels are a starting source, not asserted sufficient for an LLM. Compare Greedy/HEFT/PEFT/MSLBL-adapted/ten-rule/frozen teacher; teacher is a strong quality/time reference, not an OPT claim.

## Stage C: separate action-space change

Only after StageB, consider priority r[i] plus task-pool a,w and optional opportunity-value terms. Compare GNN versus LLM under both fixed and learned order. Same expanded supervision for all encoders; currentfixedHEFT teacher labels alone are not a source of better order targets. Use exact categorical conditional action probabilities for PPO, with environment-only updates between decisions. Define ready-set selection and insertion semantics before execution, preserve nonpreemption/cumulative capacity/hardbudget/fixedlag.

## Measurements and rejection

- Makespan gains versus strong methods; paired same-instance BC+PPO versus BC; fixed validation selection,seed-level and graph-cluster uncertainty.
- Total/model trainable parameter counts, training trajectories and walltime; GPU memory; token count; preprocessing/forward/decoder/end-to-end latency.
- Exact feasibility checks and exactlyone full model call plusone constructed schedule per final output. Include suffix query visibility under causal masking and ID-permutation consistency checks.
- A future practical gain target may be chosen (for example2%against a locked strong comparator), but this is not a conference acceptance threshold and is not locked now.
- Negative or unstable gains, infeasible outputs, extra model calls or a simple-controller match block scale-up. Do not select a winning subset of TEST instances.

Reference:[role/input audit](../../literature/llm_roles_20261015.md).
