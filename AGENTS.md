# Agent handoff — read this first

Owner:jiayiH123. Frozen evidence snapshot2026-10-15; active plan updated2026-10-18. This repository is an active research handoff, not a completed new Attention/BOPO or LLM method.

**Current priority: read `docs/SERVER_EXPERIMENT_PLAN_20261018.md` and `docs/HANDOFF_PROMPT.md`. Implement and evaluate small structured Attention plus complete-feasible-schedule preference training, using ReSched/BOPO as attributed sources. The old LLM-first plan is deferred. User permits another computer's GPU resources; exact device typed “A00” remains to be checked. Do not start by downloading language-model weights.**

## Objective and task

Minimize global DAG makespan with communication delays, cumulative resource capacity and hard additive execution-cost budget. Preserve one complete policy-network forward and one final schedule construction. Target venueIWQoS. First reproduce frozen CPU GNN/BC/PPO and adapted baselines, then prepare public-workflow development data and implement new device-aware policy/training modules. Inspect hardware before choosing batches; no server credentials or particular GPU model are supplied here.

Read docs/SERVER_EXPERIMENT_PLAN_20261018.md,docs/HANDOFF.md,docs/HANDOFF_PROMPT.md and docs/problem_model.pdf. `docs/LLM_INTERFACE.md` is a retained future option. Task is metric-driven experimentation, not restarting a broad search for an unrelated research topic.

## What is established

Original main data is synthetic20/40/80-task,3pool;800TRAIN/216VAL/432originalTEST. Laterknown_cycle24_test has432configs/48basegraphs and has been inspected. Training teacher is a ten-rule mapping portfolio plus64local proposals, not a neural model or solver-optimal labels.36540TRAINactions. BC is supervised action imitation; current PPO also retains.02teacheranchor.3seed checkpoints are provided.

Current retained14946parameter LOCALseed11 is a neural reference, not the bestknownoverallalgorithm. It beats weakbaseline means but stronger budget-aware methods/teacher beat it on the knowncycle24fold. Stable RL-only improvement and surpassing teacher remain unshown. Prior experiments with genericedgefeatures,budgetfeatures,prefixbaselines,widerscores andfixedprefixconsequence supervision mostly failed; read archive/findings.md and evidence before repeating.

Earlier user explicitly targeted surpassing teacher, stableRLincrement andadaptedbaselines. Later conversation clarified teacher-superiority is not logically necessary for a quality/runtime contribution; that re-ranking was AI advice, not permission to conceal teacher results. Keep all strong controls and report bothquality/runtime.

## Methodological rules

- Budget/resource/dependency/communication constraints are authoritative simulator checks, not prompt suggestions.
- GNN/LLM must receive equivalent physical information for representation comparisons. Separate backbone changes from learnedtaskorder/expandedsolution spaces.
- Ordinaryautoregressive textgeneration oroneAPIrequest doesnotimplyoneforward. Instrument the actualbaseforward count. Node/pairquery positions must see thecompleteinput prefix undercausalattention.
- Existingteacherlabels arefixedHEFTorder. Newprioritylearning needsnewTRAINsupervision orRL, equally available toGNNcontrols.
- First new comparison keeps fixedHEFT construction order, LOCAL decoder and score ranges across GNN/Attention x PPO/BOPO. Same architecture shares its BC start; different architectures share data/teacher rather than incompatible weights. Account historical GNN warmstart work.
- CPU reference does not become CUDA code by adding a flag. New Attention/BOPO/CUDA code is not supplied yet; executable additions are hardware preflight and known-public-smoke preparation.
- Each candidate must replay its own feasible sets, EFT and budget history for log probability. Training multiple candidates is allowed; deployment remains one deterministic construction. Actual rollout B and retained candidate K are separate counts.
- NevertrainfromTESTlabels orselectbyTEST. Reserve freshgraphdisjoint confirmation aftersettingsselection; allbundled tests are known.
- Baselines:Greedy,Randommean5,HEFT,PEFT,MSLBLadapted,tenruleportfolio,andfrozenteacher asquality/time reference. Distinguish paperadaptations fromauthorcoderuns.
- Report BC versusBC+PPO onpairedinstances,>=3seeds forclaims,confidence intervals andnegativecases. Reporttotal/trainableparameters,trainingwork,endtoendlatency andmemory.
- Freezeprotocolandcommit beforeexperiments. No hiddenpostprocessingsearch orresult-affectingcomponents. Newoutputs inignoredruns/,nooverwritingfrozencheckpoints/data/evidence.
- If autoresearch/brainstorming/ml-training-recipes are installed,use them where relevant; never claim a skill was applied when unavailable. ARA recording is sessionepilogueonly. No persistentbackgroundagent is supplied bythisrepo.
- Do not uploadmodelweights,tokens,.env,privatekeys orservercredentials. User authorizedthisresearchrepository; additional remoteGPUspending is not implied.

## First verification

`python scripts/server_preflight.py --output runs/preflight.json`; `python scripts/check_bundle.py`; then smoke and fullknownfold evaluation perREADME. Install compatible dependencies in an isolated environment; use `--require-cuda` after installing the right wheel. `scripts/prepare_public_smoke.py` prepares six historically inspected public traces, never fresh confirmation data. `scripts/train_reference.py` remains CPU reference training. New implementation tasks and actual commands are in the current server plan.
