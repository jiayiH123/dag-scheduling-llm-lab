# Independent-seed development replication, not public confirmation

Adaptive choice: follow-up seed11 BC ranks edge_attention(.853677) ahead of
edge_gnn_ln(.876920) and edge_gnn(.998554) on the216knownVAL configurations.
These BC rankings were observed before this protocol. Seed11's optimization
results are still being completed; no positive conclusion is assumed.

Repeat the same Attention BC400/batch8/LR1e-3 and PPO/SIL/pairwise/BOPO-adapted
RL40/batch8/B8/K4/LR1e-4 at seeds22 and33. Same800TRAIN/216knownVAL,fromscratch,
zeroheads,step0eligible,exactfixed-orderdecoder. Do not alter configurations after
seeing seed22. This supplies3developmentseeds with seed11, not an independent
architecture-selection test or a public-workflow result. Strong controls and
per-seed tail/paired intervals remain mandatory. Keep every objective and last
checkpoint; do not summarize only the bestseed or bestofB.

No further architecture expansion is required in this bounded session. Server
public-data training remains a later stage with a new frozen manifest/protocol.
