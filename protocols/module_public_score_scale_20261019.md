# Public-score calibration, adaptive after explicit TRAIN representability audit

Observed on the frozen publicTRAIN only:411/4992teacher decisions in19/64configs
cannot win under any independently chosen a,w in[-8,8] with original EFT offsets.
For teacher choicej and competitori the greatest possible scoremargin is
16+8*(phi_j+phi_i)+offset_j-offset_i; a negative value proves impossibility at
that teacher prefix. This diagnoses our boundedscore interface,not a new theorem
about all DAG schedulers,and does not prove another mapping cannot beat teacher.

Test one change: offset_i becomes (oldoffset_i−max(oldoffsets)) /
max(1,max(oldoffsets)−min(oldoffsets)). It preserves zero-head EFT ordering/ties
and bounds the relative prior to[-1,0]. Hardphysics,HEFTorder,a/wcap,feasibility,
teacher mapping,features anddeployment1forward/1decode remain unchanged. No
extra candidate search occurs in deployment. This is analytic score calibration.

Use the same public64TRAIN/32VAL,dataSHA and200BC/20RL seed11 recipe as the
originalpublicpilot. Both edge_gnn_ln andedge_attention; objectivesPPO,BOPO,
PPO+BC-KL1. Labels are replayed from the same TRAINteacher mappings under new
score offsets; the underlying teacher task schedule must stay exactlyidentical.
Preserve old source data rather than rewriting labels. Each optimizer branches
from its own new BC. Originalpublicpilot is the earlier completed control.

Required checks: zeroheads schedule equals originalHEFT; forcedteacher replay
identity on all64TRAIN; probabilistic replay at float64; savedphysicschecks;
all originalpublic baseline makespans identical under this decoder adaptation.
Save selected/last/negativecases. Primary question: does this remove artificial
BCcompression restrictions and improve publicVALmakespan? Do not declare method
validation or novelty from a single4graphVAL. ReservedTESTstaysunread.
