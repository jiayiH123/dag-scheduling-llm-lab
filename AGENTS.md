# Agent handoff — read this first

Owner:jiayiH123. Research snapshot2026-10-15. This repository is a portable handoff of an active, unpublished research project, not a completed LLM method.

## Objective and task

Minimize global DAG makespan with communication delays, cumulative resource capacity and hard additive execution-cost budget. Preserve one complete policy-network forward and one final schedule construction. Target venueIWQoS. First reproduce frozen GNN/BC/RL and adapted baselines, then implement a small LLM structured policy using locally available server model weights. Ask for the actual model path/hardware only after inspecting the server; do not assume a model or API key exists.

Read docs/HANDOFF.md,docs/LLM_INTERFACE.md,docs/HANDOFF_PROMPT.md and docs/problem_model.pdf. Task is metric-driven experimentation, not restarting a broad search for an entirely unrelated research topic.

## What is established

Original main data is synthetic20/40/80-task,3pool;800TRAIN/216VAL/432originalTEST. Laterknown_cycle24_test has432configs/48basegraphs and has been inspected. Training teacher is a ten-rule mapping portfolio plus64local proposals, not a neural model or solver-optimal labels.36540TRAINactions. BC is supervised action imitation; current PPO also retains.02teacheranchor.3seed checkpoints are provided.

Current retained14946parameter LOCALseed11 is a neural reference, not the bestknownoverallalgorithm. It beats weakbaseline means but stronger budget-aware methods/teacher beat it on the knowncycle24fold. Stable RL-only improvement and surpassing teacher remain unshown. Prior experiments with genericedgefeatures,budgetfeatures,prefixbaselines,widerscores andfixedprefixconsequence supervision mostly failed; read archive/findings.md and evidence before repeating.

Earlier user explicitly targeted surpassing teacher, stableRLincrement andadaptedbaselines. Later conversation clarified teacher-superiority is not logically necessary for a quality/runtime contribution; that re-ranking was AI advice, not permission to conceal teacher results. Keep all strong controls and report bothquality/runtime.

## Methodological rules

- Budget/resource/dependency/communication constraints are authoritative simulator checks, not prompt suggestions.
- GNN/LLM must receive equivalent physical information for representation comparisons. Separate backbone changes from learnedtaskorder/expandedsolution spaces.
- Ordinaryautoregressive textgeneration oroneAPIrequest doesnotimplyoneforward. Instrument the actualbaseforward count. Node/pairquery positions must see thecompleteinput prefix undercausalattention.
- Existingteacherlabels arefixedHEFTorder. Newprioritylearning needsnewTRAINsupervision orRL, equally available toGNNcontrols.
- NevertrainfromTESTlabels orselectbyTEST. Reserve freshgraphdisjoint confirmation aftersettingsselection; allbundled tests are known.
- Baselines:Greedy,Randommean5,HEFT,PEFT,MSLBLadapted,tenruleportfolio,andfrozenteacher asquality/time reference. Distinguish paperadaptations fromauthorcoderuns.
- Report BC versusBC+PPO onpairedinstances,>=3seeds forclaims,confidence intervals andnegativecases. Reporttotal/trainableparameters,trainingwork,endtoendlatency andmemory.
- Freezeprotocolandcommit beforeexperiments. No hiddenpostprocessingsearch orresult-affectingcomponents. Newoutputs inignoredruns/,nooverwritingfrozencheckpoints/data/evidence.
- If autoresearch/brainstorming/ml-training-recipes are installed,use them where relevant; never claim a skill was applied when unavailable. ARA recording is sessionepilogueonly. No persistentbackgroundagent is supplied bythisrepo.
- Do not uploadmodelweights,tokens,.env,privatekeys orservercredentials. User authorizedthisresearchrepository; additional remoteGPUspending is not implied.

## First verification

`python scripts/check_bundle.py`; then smoke andfullknownfold evaluation perREADME. Onserver install compatibledependencies inisolatedenvironment. `scripts/train_reference.py` isreferenceCPUtraining; archivedoriginalscripts arehistorical,notportableentrypoints. LLMimplementation isnewwork,notalreadydone.
