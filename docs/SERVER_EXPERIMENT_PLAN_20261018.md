# 服务器实验交接：通信感知异构 DAG＋结构化 Attention＋完整解偏好训练

更新：2026-10-18。对象：另一台电脑上的实验执行者/agent。

**这是当前优先方案，优先级高于2026-10-15的LLM交接说明。先做小型图网络与训练方法，不下载Qwen/Llama。用户报告另一台电脑有“A00”资源，暂按A100档规划；实际型号、显存、卡数、CPU和内存必须先盘点。**

## 1. 最终目标与当前事实

目标是在通信延迟、累计资源容量、硬总预算下，最小化每个DAG的全局makespan；部署严格限制为一次完整策略网络forward＋一次调度构造。目标会议IWQoS。

继续追踪用户原有目标：超过模仿教师；使后续学习相对BC有稳定增量；超过相关适配baseline。质量—时间折中也可研究，但不得把它当成用户已经放弃这些目标。

当前已实现并在原机验证：物理模拟/可行性检查、Greedy/HEFT/PEFT/MSLBL适配/十规则组合/RandomMean5、教师程序、14946参数GNN、固定HEFT构造顺序的解析解码器、CPU参考BC/PPO、合成数据与3种子检查点。当前神经模型在已知合成集上平均胜过弱对照，但弱于MSLBL适配、十规则组合和教师；独立RL增量尚未稳定。

教师是十规则映射重放＋64次1/2任务改池尝试的离线程序，不是网络，也不是OPT。800训练配置提供36540动作标签。当前参考PPO是actor-only PPO-clip，没有critic/GAE；含.02教师交叉熵。旧200BC/300PPO与180秒上限只作迁移复现，不是主实验训练预算。

| 内容 | 本次交接状态 |
|---|---|
| 冻结CPU参考、物理约束、baseline、旧数据 | 已实现，按第4节复核 |
| 硬件盘点脚本、公开工作流known-smoke下载/转换 | 已提供，见第4–5节 |
| 主实验公开TRAIN/VAL/TEST清单及平台 | 需在服务器选择并冻结；不能把known-smoke当新测试 |
| device-aware GNN、结构化Attention、BOPO训练、GPU批处理 | **待实现**，接口和验收见第7–9节 |
| ReSched/BOPO/DAG-CAI-DR原论文数值复现 | **尚未完成**，代码公开不等于已复现 |
| 新方法效果、超过教师、显存与运行时间 | 未测量，不能先写结论 |

## 2. 仓库、论文、数据与版本

本项目：[jiayiH123/dag-scheduling-llm-lab](https://github.com/jiayiH123/dag-scheduling-llm-lab)。首次clone，已拉取过则在工作区无冲突时`git pull --ff-only`。使用HTTPS；公开仓库无需登录，若仍是私有则在该机器独立认证。

| 来源 | Git地址/文档 | 用途及边界 |
|---|---|---|
| ReSched，ICLR2026 | [代码](https://github.com/XiangjieXiao/ReSched)／[论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html) | task–machine交互/Attention参考；原版逐步forward、串行机器，不能原样当我们同场景baseline |
| BOPO，ICML2025 | [代码](https://github.com/L-Z-7/BOPO)／[论文](https://proceedings.mlr.press/v267/liao25a.html) | 完整解偏好训练参考；原版JSP/FJSP/TSP，无完整硬预算通信场景 |
| DAG-CAI-DR，NeurIPS2025 | [代码](https://github.com/9erAlpenjaeger/DAG-CAI-DR)／[论文](https://papers.nips.cc/paper_files/paper/2025/hash/dc16950e510bcd573783f5c2d7d8cc31-Abstract-Conference.html) | 近期one-shot DAG外部对照；原版学优先级，异构实验由EFT选pool，无完整硬货币预算 |
| WfInstances | [真实执行JSON](https://github.com/wfcommons/WfInstances)／[浏览器](https://wfinstances.ics.hawaii.edu/) | 任务图、runtime、输入输出文件及执行记录；不是全部异构候选池的反事实性能矩阵 |
| WfCommons/WfGen | [代码](https://github.com/wfcommons/WfCommons) | 可选训练数据生成；基于真实应用模式的合成，不能称新实测trace |
| HDSE，NeurIPS2024 | [代码](https://github.com/LUOyk1999/HDSE) | 后续结构bias备选，首轮不叠加 |
| PyTorch | [官方安装选择器](https://pytorch.org/get-started/locally/) | 根据服务器驱动安装兼容CUDA wheel，不更换系统驱动 |

固定commit及角色在[EXTERNAL_SOURCES_20261018.json](EXTERNAL_SOURCES_20261018.json)。以该清单为读取起点，若上游后续修复另开分支记录差异。第三方源码不要直接混入我们的vendor。

```bash
mkdir -p runs/third_party
git clone --filter=blob:none https://github.com/L-Z-7/BOPO.git runs/third_party/BOPO
git -C runs/third_party/BOPO checkout --detach 2739fbfe39a478755173c892b80c4b06f9f59b05
git clone --filter=blob:none https://github.com/XiangjieXiao/ReSched.git runs/third_party/ReSched
git -C runs/third_party/ReSched checkout --detach 4d1fda8ab4d78d1cd9664de28f3f77d88620f309
git clone --filter=blob:none https://github.com/9erAlpenjaeger/DAG-CAI-DR.git runs/third_party/DAG-CAI-DR
git -C runs/third_party/DAG-CAI-DR checkout --detach 1a2aacc3fc44d8d8f1ef779488a39cab80abf991
```

这些命令只取源码，不执行作者训练。作者项目分环境安装。DAG-CAI-DR的Linux/Python3.9/Torch1.13旧环境不宜覆盖主环境；原脚本默认多任务/GPU1，先单配置。BOPO公开训练入口有DEBUG/注释训练调用等静态审查点，先读并记录修复；不能承诺克隆即运行。所有借鉴遵守各自许可并引用论文。

## 3. 要准备什么

- Linux服务器及可使用的GPU，优先单卡起步；确认实际型号/40或80GB等显存，不猜“A00”的含义。
- Python3.11隔离环境、与驱动兼容的PyTorch、NumPy、OR-Tools；`requirements.txt`只包含后两者。无需LLM权重、API key、DGL或全部作者项目依赖。
- 多核CPU和足够RAM用于日历搜索；初始4个环境worker，再profile扩展。单GPU可先顺序跑种子，不要求多卡。
- 可写实验盘，预留几十GB量级的代码、环境、数据和检查点空间后按实际增长管理；原始trace逐JSON下载，勿先clone整个WfInstances大仓库。
- GitHub/论文/数据下载访问。若网络不可达，先记录具体URL错误并准备离线文件，不改数据来源、不假报已下载。

资源是规划，不是实测保证：开发50–200任务/3–8pool、2–4层/hidden128、实际候选8–16；通过后扩100–500任务/8–16pool、候选32–64。原ReSched用单A40、BOPO用TITANXp；采用小型网络时，A100是合理起点。训练时长须实测吞吐估计，不能引用旧24秒试跑。

## 4. 首先运行：环境盘点与冻结参考复核

以下是**当前真实可执行命令**，在仓库根目录运行。每次选择新输出目录，脚本拒绝覆盖。

```bash
git clone https://github.com/jiayiH123/dag-scheduling-llm-lab.git
cd dag-scheduling-llm-lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

然后从PyTorch官方选择器安装本机兼容CUDA版本；该步骤取决于驱动，不在文档中硬编码一个CUDA wheel。不要更改系统CUDA/驱动，也不要自动安装`requirements-llm.txt`。

```bash
python scripts/server_preflight.py --require-cuda --output runs/preflight_20261018.json
python scripts/check_bundle.py
python scripts/evaluate.py --limit 8 --teacher --output runs/reference_smoke.json
python scripts/evaluate.py --data data/known_cycle24_test.jsonl.gz --teacher --output runs/reference_known432.json
```

`preflight`检查的是该解释器的CUDA可用性；即使失败也保存盘点。当前`evaluate.py`仍执行CPU参考，服务器有GPU不意味着该脚本会使用GPU。这是有意复核旧结果。

known432种子11的平均逐实例相对收益应接近：Greedy20.966168%、HEFT15.582848%、PEFT17.557653%、RandomMean5 46.359614%、MSLBL−3.503358%、Portfolio−10.091474%、Teacher64−12.625171%。负值表示学生更差。超出合理数值容差先查Torch版本/物理口径/Random种子，不继续用差异当新提升；原置信区间bootstrap seed可能不同。

可选重放旧训练（不作为主实验）：

```bash
python scripts/train_reference.py --seed 11 --output runs/reference_replay_seed11
```

该脚本没有`--device`、`--epochs`或`--candidates`参数，固定旧配方且读取早期GNN BC warmstart。不要伪造CUDA命令；新训练器必须另写。若Torch/模拟器在两环境，可对evaluate指定`--torch-python`，对train_reference指定`--sim-python`。

## 5. 已提供的公开数据smoke入口

```bash
python scripts/prepare_public_smoke.py --download --raw-dir runs/public_smoke_raw --output runs/public_smoke.jsonl.gz
python scripts/evaluate.py --data runs/public_smoke.jsonl.gz --split val --limit 4 --teacher --output runs/public_smoke_reference.json
```

下载固定WfInstances提交的6个完整JSON，校验原SHA256，生成2链路×2预算×6trace=24条兼容记录。不需要下载整个数据仓库，也不执行工作流命令。已有原始文件时省略`--download`，按manifest文件名放入`--raw-dir`。

**这6条Montage58、1000Genome52、Cycles67、Epigenomics41/73、SRA64均在原研究中看过，只用于工程smoke/回归，不是新盲测，也不是主训练数据划分。**记录使用`val_public_smoke_`前缀仅为兼容现有评测过滤器，无训练教师标签。不要把该文件传给旧训练脚本期待得到TRAIN。

来源、原SHA、许可链接与已知范围在[data/public_smoke_sources.json](../data/public_smoke_sources.json)。数据转换假设在`daglab/workflow_trace.py`：实测图/runtime/filebytes保留；统一速度、slot需求、容量、价格、带宽与预算是模拟参数，不冒充实测。

本次交接已真实下载6条raw并跑通全部24配置；旧LOCAL seed11对HEFT平均收益16.659%，但对MSLBL为−17.051%、Portfolio为−19.884%、Teacher64为−21.715%。这只是旧模型工程回归，不能作为新方法或新盲测的效果；逐实例记录见[交接smoke结果](../evidence/public_smoke_reference_20261018.json)，准备检查见[验证记录](../evidence/SERVER_HANDOFF_CHECKS_20261018.json)。

## 6. 主实验公开数据：先冻结划分，再生成变体

1. 从WfInstances选择新执行实例，读取应用README及schema。先在不查看调度效果的情况下确定样本和来源commit；旧smoke不进入最终未见测试。
2. 先按基础trace分TRAIN/VAL/TEST，再派生预算/网络变体。相同基础trace的所有配置、生成子图/扰动均在同fold。若研究跨应用泛化，另留完整应用族；不同trace同模板不自动代表新结构。
3. 保存manifest：源URL/commit、原SHA、应用、底层ID、split、任务数、平台版本、预算模式、是否合成、是否已查看。基础实例数与配置数分别统计。
4. 训练样本不足时可用WfCommons/WfGen从**训练来源**校准生成器；记录版本/recipe/seed，生成物标合成。不可用TEST统计调整生成器后再称零样本迁移。
5. 统一补平台：每池speed/capacity/price，带宽矩阵，标量latency，任务需求，计时/费用单位；相同配置用于全部baseline。实测CPU利用率/峰值memory不能直接等同资源保留需求，需解释映射。
6. 新公共TRAIN可重新运行教师程序生成GT；VAL/TEST不写训练标签。Teacher64若作测试质量参照，只作诊断。新教师数据也不保证OPT。

数据契约与现有函数：

```text
record = {key, instance, graph, reference}
TRAIN additionally = {teacher, teacher_decisions}
key starts with train / val / test
reference = same-physics HEFT makespan
graph = daglab.reference.graph_features(instance)[0]
```

仅对TRAIN调用教师；函数返回三个值，需要解包后保存，不能把整个tuple当成调度方案：

```python
from daglab.reference import CachedBudgetDecoder, teacher

solution, decisions, seconds = teacher(instance, CachedBudgetDecoder(instance))
record.update(
    teacher=solution,
    teacher_decisions=decisions,
    teacher_seconds=seconds,
)
```

`daglab.io.read_records`按key前缀过滤，没有独立split字段。`daglab.sim_server`直接使用缓存graph，修改输入表示必须同步生成/加载新表示。模拟instance需兼容已有schema；可参考`data/reference_dataset.jsonl.gz`的一行及公开转换器。必须有`seed/latent_id/family/resource_mode/budget_mode/target_ccr`等评测元数据；真实trace的target_ccr可以是null，不能虚造为测量值。

费用为整数`duration*cost_rate`；总预算不得低于所有任务最小可行费用之和。现有预算档是`Bmin+eta*(Bfast-Bmin)`，eta=.25/.6/1；25%指升级余量，不是总预算的25%。新平台价速关系改变后，eta1不一定真正不限制预算；须验证或改称eta1，真nonbinding可另用全部候选最大费用之和。

## 7. 模型与解码：首轮保持一个动作空间

固定HEFT构造顺序；它不等于串行执行顺序，SGS可并行和插入空档。依赖、通信、累计容量、硬预算的权威实现是vendor内模拟器，不替换成ReSched的串行machine模型。当前通信允许重叠、无链路争用；资源为标量，首轮不加路由/拥塞或多维资源。

Attention设计要求：

- task节点：计算量、需求、DAG方向/分叉汇合及数据量；pool节点：速度、容量、价格；task–pool关系：执行时间/成本/兼容性；pool图：带宽/时延；全局：初始预算。
- 借鉴ReSched任务self-attention＋pool cross-attention，不照搬job-chain RoPE或“四个特征充分”的证明。若做结构bias，明确有向DAG语义。
- 优先task/pool/cross分解，不把全部NP对直接做(NP)²全连接。仍可输出全部NP个评分。
- 每实例输出node-major`[n*k,2]`的a,w（或batch等价形状）；沿用`8*tanh(raw/8)`范围及LOCAL解码。score=`a+w*phi−EFT/scale`，phi是局部静态升级费用比例，剩余预算通过动态reserve mask生效，EFT由各条轨迹日历/父节点分配决定。
- CUDA化需统一tensors/device，不能只`model.to(cuda)`：旧actor及conditionals内部直接创建CPU tensor。新增GPU版文件，保持冻结vendor/checkpoints/data不变。
- 首轮dropout=0，避免rollout与重放概率因随机层不一致。对所有模型记录原始物理信息差异；若Attention额外获得信息，给GNN同等输入或补信息消融，不能归因纯结构。
- 解码器仅接收缓存数值输出，不接收可反复调用的模型对象。推理forward计数与完整encoder实际执行次数一致，不把循环藏进一次外层API调用。

## 8. 训练：BC后比较PPO与BOPO适配

先分别训练GNN-BC、Attention-BC。跨架构不能加载同一权重；同架构的PPO/BOPO必须从同一BC检查点出发。两架构使用相同TRAIN/教师与可比训练预算。旧GNN有额外warmstart，需作为历史参考或另做公平从零BC控制，不悄悄给一边更多训练。

首轮每实例实际采样B=16（1greedy+15随机），保留K=8用于配对；这是试验默认值，不是BOPO原版复现。BOPO原文JSP B256/K16，FJSP B128/K16，B是真实解码数，K是筛选数。不要只报告K。

每个候选保留自己的`available/selected/offsets/budget_local/action`等历史；用自己的mask、EFT和预算特征重算logprob。原函数见`scripts/train_reference.py::conditionals`，新device-aware版须逐步对照。给定动作历史，物理状态由SGS确定，重放不需每步再调NN。

BOPO参考：[JSP/sampling.py](https://github.com/L-Z-7/BOPO/blob/2739fbfe39a478755173c892b80c4b06f9f59b05/JSP/sampling.py)的`SROLoss`与`sample_training_pair`。完整轨迹使用平均动作logprob；最优方案与其他保留方案配对，用makespan比值调节成对logistic损失。只在同instance/platform/budget内比可行方案；ties记录并处理，不能把不可行但快的方案当赢家。

借用loss不意味着完全复现BOPO的hybrid rollout。名称应注明BOPO-adapted/inspired，记录直接移植与领域改造。PPO与BOPO目标本来不同；统一候选数、初始化、可比调参工作，报告优化步/epoch/wall time。若保留.02teacher anchor，记录各目标的归一化和梯度相对尺度，必要做共同anchor消融，不能把去掉正则的收益说成新模块收益。

## 9. 执行顺序与验收

[configs/attention_bopo_pilot.example.json](../configs/attention_bopo_pilot.example.json)是**待实现新trainer的配置契约**，当前没有脚本消费它。不要把它传给旧train_reference。

| 阶段 | 工作 | 交付/继续条件 |
|---|---|---|
| P0 | 第4节环境与参考复核 | preflight、bundle PASS、旧均值/约束一致；有差异先解决 |
| P1 | 第5节公开smoke＋主数据清单/平台/划分 | 原SHA、24已知smoke通过；新数据无跨fold底图泄漏；固定manifest |
| P2 | device-aware模型/轨迹概率/硬约束验证 | 零评分恢复同物理HEFT；CPU/GPU同动作历史logprob一致；每个保存调度通过检查；forward=1 |
| P3 | TRAIN/VAL候选信号诊断 | B8/16的唯一映射比例、ties、best-of-B相对greedy空间；无候选改善先查探索/可达性 |
| P4 | 各架构BC，四臂小试 | GNN/Attention×PPO/BOPO，先seed11；记录相对各自BC的确定性VAL曲线，不凭pair loss判断有效 |
| P5 | 额外控制与主试验 | 加best-sample自模仿、无质量加权pairwise；候选/训练预算匹配；至少11/22/33三种子 |
| P6 | 冻结后公共确认 | 锁代码/平台/数据/超参，再用新未见TEST和应用外推，保留全部负例和强baseline |

baseline至少保留Greedy/RandomMean5/HEFT/PEFT、MSLBL适配、十规则组合、Teacher64质量—时间参照。DAG-CAI-DR先在作者环境复现一项小设定，再做统一物理模型的预算适配；两个结果分开写。更慢搜索可比较质量—耗时，不要求击败无限搜索；同样快而质量更好的方法必须正面报告。

新代码建议路径（**任务清单，不是已存在入口**）：`daglab/attention_actor.py`、`daglab/trajectory_policy.py`、`daglab/preference_training.py`、`scripts/train_policy.py`、`scripts/evaluate_policy.py`。先实现CPU/GPU概率一致的小验证，再做训练。输出分`runs/<run_id>/`，冻结计划另存`protocols/`并在训练前git commit。

开发筛选不能靠换TEST找好结果。若BOPO不优于普通pairwise/自模仿，或Attention不优于信息/容量控制，不把它包装为独立贡献；也不因一次失败就否定整类方法。预算/种子扩展按VAL曲线及吞吐决定，不无上限堆规模。

## 10. 指标、记录与结果怎么交回来

- 主指标：每实例绝对makespan；成对收益`100*(C_baseline-C_policy)/C_baseline`；VAL固定使用平均`C_policy/C_HEFT`选checkpoint。不同单位/工作流不要只拼原始总时间，原始C也要保存。
- BC后增量对**同架构同解码器**的BC算。约束违例、成本/B、forward次数、部署候选数必须记录；硬约束违例不能靠罚分平均掉。
- 报告胜/负/平、最坏退步、预算/通信/规模分层。bootstrap按底层trace分组，不把预算变体当独立图；分别报告种子和汇总。
- 总推理耗时包括特征/拓扑预处理、CPU–GPU拷贝、模型、解码。CUDA计时同步；warmup单独记录；baseline同硬件/口径。一次forward不自动等于最快。
- 多候选选最好只作训练/空间诊断；主部署指标必须greedy一次forward一次decode。另报best-of-B时明确额外构造数与时间。
- 每run保存metadata、完整配置/哈希、代码SHA、上游SHA、数据manifest、训练曲线、逐实例评价、计数器、最佳VAL检查点、负例。实际候选数/完整解码数/NN calls/optimizer steps/wall time分别统计。

推荐输出结构：

```text
runs/<run_id>/
  preflight.json  config.json  metadata.json  protocol.md
  data_manifest.json  upstream_commits.json
  training_curves.jsonl  evaluation_rows.jsonl  summary.json
  validation_selected.pt  failures.jsonl
```

metadata至少含架构/训练目标/各自BC起点、seed、B与K、初始及实际更新数、数据哈希、物理配置、总/可训练参数、峰值显存、阶段耗时、测试是否已查看、forward计数方法。配置里不要带账号token或服务器凭据。实验大文件留`runs/`，挑选可审计汇总后再提交。

如果安装了autoresearch、ml-training-recipes，按需使用；不要为缺skill停工或声称使用未安装技能。ARA仅在本次工作完成后记录真实事件；没有ARA技能也要保留同等日志。每次交回：做了什么、与哪些控制比较、原始结果路径、是否有RL/偏好增量、失败在哪里、下一步根据什么决定。不得先承诺论文录用或超过教师。

最短启动提示词见[HANDOFF_PROMPT.md](HANDOFF_PROMPT.md)。历史LLM接口保留在[LLM_INTERFACE.md](LLM_INTERFACE.md)，只有当前小网络路线验证后另行考虑。
