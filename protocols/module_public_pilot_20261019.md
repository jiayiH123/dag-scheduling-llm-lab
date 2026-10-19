# Public-data development pilot, selection locked before downloads/results

User prioritizes architecture and paper draft and explicitly asks to move to the
public workloads used in related papers. Stop expanding old-synthetic sweeps.
Use pinned WfInstances/Pegasus measured traces,not proprietary One-Shot graphs.
Source paths/Git blob hashes/splits are in data/module_public_sources_20261019.json.
EightTRAIN,fourVAL;threeTESTpathsreserved but not downloaded/evaluated. Exclude
the old6knownpublicsmoke traces. Check normalized task-name-independent directed
topology fingerprints across TRAIN/VAL and fail on duplicates;do not select by
policy quality. Same-application templates remain related;no cross-family
generalization claim follows from a trace/scale split.

Measured DAG/runtime/filebytes retained. Simulate3or6pools with capacity2/unit
taskslot demand;relative speeds[1,2,4]tiledfor6pools,prices[1,4,16]tiled,
same3-type linkmatrix extended to6pools. Same-type distinctpools use the slowest
base link rather than zero communication;onlysamepooliszero. Two networkprofiles
and eta.25/.6 give8configs/trace,64TRAIN/32VAL. Parametersaredeclaredsimulation,
not measured provider hardware. Allvariantsstaywiththeirsource split.

TRAINteacher only: unchangedtenrule+64proposals. VALteacher only as quality
reference after training. No TESTlabels. One seed11 for this initial public
development screen;do not expand seed count in this phase.

Models: edge_gnn_ln,edge_attention;hidden32/layers2,fromscratch,zeroheads.
BC200updates,batch4,LR1e-3,evaluateevery50;each ownBC thenPPO,BOPO-adapted,
PPO+BC-KL1 at20updates,batch4,B8/K4,LR1e-4,teacheranchor.02. Both architectures
are run rather than dropped by adaptive selection. FullVALselectscheckpoints,
includingstep0. Frozen source policy remains unchanged.

Numerics: retainfloat32network;usefloat64conditionalsoftmax/logprob forlarge
trace-time ranges,matching the NumPyfloat64simulator. This is a common numerical
change forallpublic arms,notanarchitectural contribution. Require own-prefix
probability/feasibility checks;log any preflight failures rather than maskingthem.

Comparisons: Greedy,RandomMean5,HEFT,PEFT,MSLBLadapted,Portfolio,Teacher64;
samephysics. Record every selected/last checkpoint,BC-relative paired outcomes,
negativecases andallbaseline results. This is a small public development pilot,
not the final paper evidence or an original-paper numerical reproduction.
