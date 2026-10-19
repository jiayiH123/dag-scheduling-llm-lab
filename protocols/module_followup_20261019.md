# Adaptive follow-up, specified after first screen and before new runs

Exploratory origin: module_lab_screen_20261019, protocol836ab82/code61bb1a8.
First screen has6BC+8optimization arms. From-scratch legacy GNN often selects
step0; explicit Attention improves BC, but its gains are not isolated from
normalization and parameter count. Preference/self-imitation sometimes lower
meanC/HEFT while paired BC-relative gains crosszero and tail regressions are large.
No external TEST or teacher-superiority claim follows.

Next question: is normalization, rather than attention, sufficient for improved
supervised fitting? Add edge_gnn_ln (LayerNorm after each directed message update),
keeping the physical inputs, head cap, decoder and from-scratch start unchanged.
Compare edge_gnn/edge_gnn_ln/edge_attention. BC400 updates,batch8, all800TRAIN,
all216VAL for selection every50. LR1e-3, seed11. This changes BC budget/data jointly
from the earlier cheap screen; comparisons within this follow-up are controlled,
not an attribution of cross-round gains to normalization alone.

Top2 BC architectures by VAL proceed to PPO/SIL/pairwise/BOPO-adapted, each from its
own common BC. RL40 updates,batch8,B8/K4,LR1e-4,evaluate every10,anchor0.02. Reduce
all objectives' LR together to test more conservative updates, not one favored arm.
Step0 and last checkpoints retained; report tails, paired gains and strongcontrols.
No guarantee of convergence; one-seed exploratory stage only. Require3seed and
public-workflow follow-up before positive claims. Do not promote on VAL alone.

Also improve provenance: preserve endpoint/selected metrics and candidate counts;
add gradient-scale diagnostics for primary objective versus common teacher anchor.
Loss scaling differs between objectives and must not be hidden.
