# BC-policy KL as a training-only tail-regression control

Adaptive hypothesis, locked before KL runs: current selected Attention/PPO gains
are small, with rare severe BC-relative regressions. Test whether a distributional
trust-region to the frozen same-architecture BC reduces this without eliminating
all improvement. This is standard policy regularization, not a new RL algorithm.

Arms: Attention PPO control, PPO+BC-KL coefficient1, PPO+BC-KL coefficient10;
seeds11/22/33; same from-scratch400BC/40RL budget,800TRAIN/216knownVAL,batch8,
B8/K4,LR1e-3BC/1e-4RL,teacherNLLanchor.02. All three start from the exact same
BC perseed. Repeated BC/PPO are controls/reproductions,not counted as new gains.

At every candidate's own prefix compute the complete feasible-pool distributions
for current and frozen BC policies, using that candidate's mask/EFT/budget values.
Add mean KL(current||BC) over decisions,candidates,instances to PPO. Frozen BC
forwards occur only in training and are charged separately. Deployment still
uses one current-policy forward and one decode,with no reference fallback.

Select checkpoints with the same meanC/HEFT rule includingstep0. Report mean
paired gains,all perseedcluster intervals,p95BC-relative regression and maximum
regression. The exploratory joint gate requires at least2/3seeds with p95positive
regression at least20%smaller than PPO while meanpairedgain is no more than.25pp
worse. Do not hide any quality/tailtradeoff or choose by unseenTEST.

Both model probabilities and simulator invariants must pass checks after the
distribution refactor. Existing non-KL controls should reproduce saved weights
and scores under the same version/environment; report mismatches if any.
