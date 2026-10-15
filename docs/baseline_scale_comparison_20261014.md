# Baseline sufficiency and primary-paper experiment scales

Date: 2026-10-14. User asks whether beating Greedy/HEFT suffices for publication,
and whether related work also uses only three processors. Literature-only audit;
no new training, scheduling experiments, or method reproduction.

## Assessment and scope

Beating Greedy/HEFT is useful performance evidence. A gap between those baselines
and OPT establishes headroom on the tested instances; it does not establish a
learned method's contribution, prior-art novelty, or conference acceptance.
There is no universal requirement to beat every retrieved baseline. For this
already-studied hard-budget problem, a representative budget-aware control is
needed to distinguish a method improvement from a weak budget adaptation.
A scheduling-quality versus full inference/decode-time advantage can support a
contribution even when an unconstrained iterative search has a better makespan.
These are AI methodological recommendations, not a quoted venue baseline-count rule.

## Verified scales

| Work | Resource counts and semantics | Tasks / experimental data | Compared methods | Primary location |
|---|---|---|---|---|
| MSLBL, FGCS2017 | 128 simulated heterogeneous serial processors. Three processors occur in a worked example, not the main platform | Reported overall range10–2627; FFT scale sweep96/224/512/1152/2560; Gaussian elimination77–2627. Application-pattern simulations, not measured deployment traces | HBCS for budget/makespan; HBCS/DBCS for budget/deadline acceptance | Section5.1,5.2,5.3; Tables7,9 |
| BUDA, IWQoS2022 | 32 and64 fully interconnected heterogeneous processors, serial availability | Random task counts20/40/60/80/100; CCR.1/5/10; paper reports4000 random DAG instances and25 generated per DAG type. Also molecular dynamics structure41tasks (Fig.2) and Epigenomics100 | BHEFT,HBCS,FBCWS | SectionV-B/V-C,Fig.3/4 |
| AOM-SO, FGCS2026 (online2025) | Nine available VM TYPES; actual VM COUNTS3/6/12 in scale datasets. Independent VM-count sweep6/12/18/24/30 | Small: EP19,LIGO40,GE27,Montage18; medium: EP103,LIGO274,GE299,Montage126; large: EP523,LIGO544,GE495,Montage606. Further task-count sweep includes intermediate51/94/54/66; molecular dynamics41.15 repetitions for success rate | OSAM,MG-PRO,SO,AEFT (budget-adapted) | Section5.1,Table3/4,Section5.1.2 |
| BCWS, IEEE TSC2024 | Dynamic VM renting plus heterogeneous serverless functions; resource instance number chosen by algorithm, not fixed3. One homogeneous VM type per scenario; four VM-type settings across calibration. Each instance serial | Tasks100/200/300/400.4800 calibration configurations across parameter/type settings,90 component choices.4000 realistic-pattern comparison instances from BWA,Cycles,Epigenomics,SoyKB (WfGen); not4000 measured independent traces | CG,GRP-HEFT,SMOHEFT | SectionIII assumptions,SectionV-A/V-B; local32-page manuscript |
| GiPH, MLSys2023 | DL experiment8devices; dynamic-network experiment16–20devices. Separate traffic case includes36RSUs plus40additional edge devices and CAV-related nodes; these are different scenarios, not a single fixed network | DL:300graphs,200–300operators each compressed to40 scheduling groups; split150train/150test. General-graph experiment300graphs split equally; network-transfer test500 sampled combinations of10networks×120graphs | HEFT,Random,Placeto,RNN-Placer and variants | Section5.1/5.2/5.3. Original operator count must not be called actual policy graph size |
| WeCAN, arXiv2603.23249v2 (2026 preprint) | Main experiments3 heterogeneous cumulative-capacity resource pools, with two resource dimensions. Cross-size/pool generalization includes12pools | TPC-H30/50/100 denotes number of query DAGs per scheduling instance, with average275/459/918tasks. Computation graphs500tasks. Generalization TPC-H150/200 approx1500/2000tasks. Training query-DAG counts2464/3850/7700;10test scheduling instances per main TPC-H setting | CP,SFT,MOPNR,Tetris,HEFT,PEFT,IPPTS;PPO-BiHyb,One-Shot,ScheduleNet,HGN-Two-Stage | SectionV-A/V-F,TablesII/V,AppendixC.1/C.4 |

Sources:

- MSLBL: https://www.cs.newpaltz.edu/~lik/publications/Weihong-Chen-FGCS-2017.pdf ; full text retained in `refresh_20261012/budget2017.txt`.
- BUDA: https://xujia-njupt.github.io/xujia.github.io/Paper/2022IWQoS.pdf ; full text retained in `refresh_20261012/buda2022.txt`.
- AOM-SO: https://www.cs.newpaltz.edu/~lik/publications/Yanfen-Zhang-FGCS-2026 ; DOI https://doi.org/10.1016/j.future.2025.108118 ; full text retained in `refresh_20261012/aom2026.txt`; code https://github.com/Yanfen-Zhang/AOM-SO .
- BCWS author manuscript: https://riunet.upv.es/server/api/core/bitstreams/faaa83de-75b6-4610-bf24-26a34658c381/content ; downloaded this session to `scale_refresh_20261014/bcws2024.pdf`; code https://github.com/jinquanaccount/BCWSP .
- GiPH official paper: https://proceedings.mlsys.org/paper_files/paper/2023/file/3e3eec95971350490e37a076fdc100ad-Paper-mlsys2023.pdf ; existing text in research-root `results/communication_dataset_survey_20261012/sources/giph_mlsys2023.pdf.txt`.
- WeCAN primary full text: https://arxiv.org/html/2603.23249v2 . This is a preprint; no accepted-venue status is inferred.

## Non-equivalences

- Serial processor/VM, device, and cumulative resource pool counts are not interchangeable. A three-pool benchmark can support many simultaneous tasks per pool.
- The first four papers concern budget scheduling; GiPH and WeCAN are method/scale references, not full matches to our hard monetary-budget model. WeCAN lists communication as future work.
- Diagram/worked-example counts are not experimental platform counts.
- Paper-reported configurations, repeated stochastic runs, graph/device combinations, and independent measured DAGs are different sample units.
- LPWS full experimental parameters and GAPPO primary paper remain insufficiently verified. Do not guess their processor or task counts.
- BCWS relative-percentage deviation uses the best compared method, not proven OPT. AOM-SO's RDI comparator also must not be assumed global OPT. WeCAN's heuristic-relative improvements are not exact OPT gaps.

## AI recommendations, not executed

Keep a compact relevant comparison set, e.g. Random,Greedy,HEFT,PEFT,MSLBL-adapted,
and the available AOM-SO-inspired/model-adapted search control. State clearly which
are author-code reproductions versus adaptations, use common communication/budget/
capacity semantics, and charge complete feature/forward/decode time. Existing actor
currently derives much of its gain from BC, so BC versus BC+RL is an essential
attribution ablation when claiming an RL-specific improvement.

Current20/40/80-task,three-device data support a development pilot. A practical next
scale matrix is tasks50/100/200/500 and devices3/6/12, optionally1000tasks and24devices,
with more independent graphs and fixed-device/fixed-task-device-ratio controls.
This is a proposed coverage range, not a venue requirement or an experiment result.
No need to certify global OPT for every large instance; retain small exact tests and
honest large-instance lower/upper bounds alongside meaningful baselines.
