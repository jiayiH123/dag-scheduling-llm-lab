# 模块与公开数据开发结果（2026-10-19）

完成73个训练臂，其中6个是张量完全一致的重复控制；构造97280条训练候选完整调度。全是开发/已知VAL结果，不含独立TEST。原106冻结文件及P4 LOCAL11默认模型不变。

## 执行范围

| 阶段 | 臂数 | 训练候选 | 用途 |
|---|---:|---:|---|
| screen | 14 | 5120 | 六种BC初筛、两架构×四优化目标 |
| followup | 11 | 20480 | 充分BC、LayerNorm控制 |
| replication | 10 | 20480 | Attention种子22/33 |
| gnn_ln_replication | 10 | 20480 | 归一化GNN种子22/33 |
| bc_kl | 12 | 23040 | BC-KL控制，含六个重复臂 |
| public_pilot | 8 | 3840 | 公开数据、原评分 |
| public_scaled | 8 | 3840 | 公开数据、相对EFT评分 |

各协议先独立commit再运行，metadata保存protocol/code commit。候选计数包含确定性候选，部署只有一次forward和一次确定性构造。每条候选重放自己的mask、预算与EFT；所有materialize调度经过物理校验。

实现的模块包括旧12维/新20维信息、显式通信消息、LayerNorm、分轴Attention、OCT先验及MLP控制。训练目标包括BC、PPO、自模仿、普通pairwise、BOPO适配，另测到冻结BC的KL约束。没有扩成全部笛卡尔积。

GNN沿DAG上下游聚合；Attention在DAG消息上再做任务轴、pool轴注意力。Attention不是LLM。归一化GNN13442参数，Attention30466参数，当前未匹配容量。legacy_gnn是12866参数的新共享head控制，不是冻结14946参数模型的逐字复现。

## 原合成开发集

800TRAIN/216VAL，VAL只有24张底图。Attention三种子结果如下；C/HEFT均值与以BC作分母的配对收益是不同统计量。

| 方案 | 三种子均值C/HEFT↓ | 逐例相对自身BC平均改善 |
|---|---:|---:|
| BC | 0.85918 | — |
| PPO | 0.84424 | +0.8745% |
| BOPO适配 | 0.85048 | +0.6977% |
| PPO+BC-KL1 | 0.84827 | +0.9064% |

归一化GNN的PPO收益为+2.024%/−0.347%/+0.248%。Attention普通PPO各种子的图聚类区间均跨0。温和BC-KL降低三种子的p95退步，但平均C/HEFT略差，最坏退步仍很大；过强KL妨碍改善。BOPO没有稳定超过PPO。

最好运行可以单列为“最佳开发运行”，同时保留已有均值和负例，无须为此继续扩大旧合成扫参。这些神经结果仍落后MSLBL、portfolio及teacher。

## 公开数据实际有多少图

源：[WfInstances](https://github.com/wfcommons/WfInstances)，固定commit d7bfead08a4dbf985ebbf427b3685a6512c259e2。路径/Git blob/SHA256及许可链接见data/module_public_sources_20261019.json和证据目录manifest。

8TRAIN指8张原始DAG，任务数58/104/82/119/22/42/96/101，平均78；4VAL指另外4张执行trace，任务数103/156/233/84，平均144。每图×3或6pool×两网络×两预算=8配置，故64TRAIN/32VAL。这不是64/32张独立图。TRAIN/VAL拓扑指纹无重合，但有同应用族，不能声称跨应用泛化。3条TEST路径只预留，未下载或读取。

拓扑、runtime、文件大小来自真实执行记录；算力/价格、每池容量2、每任务需求1、网络和预算是声明的模拟假设。TRAIN运行原teacher64程序生成4992动作标签，无求解器；VAL教师只作质量参照。

## 评分尺度问题与修复

原评分a+w*phi−EFT/scale中a,w限制在[-8,8]。真实工作流的EFT量级使先验过强：TRAIN的4992教师动作中411个，即使让两个输出独立任取最有利值，也无法赢过某个竞争pool，涉及19/64配置。增大网络不能自动解决这个表达限制。之前合成数据上也发现过评分可达性问题，本次是公开数据上的检查和修复，不宣称领域首创。

教师选择j相对竞争i的最高分差为16+8(phi_j+phi_i)+offset_j−offset_i，负值证明该prefix的教师动作不可表达。新版本仅把当前可行pool的offset平移/缩放到[-1,0]，保留EFT排序、cap8、物理模型和固定HEFT构造顺序。此修复移除该特定障碍，不证明共享网络能表达任意映射。

96次zero-head/HEFT完整调度一致；64个TRAIN教师映射完整重放一致；七baseline×32配置的224个目标完全一致；float64概率重放最大误差2.22e-16。原vendor未修改，新decoder隔离实现。

## 公开数据结果：一颗种子、四张VAL图

| 架构/训练 | 原评分C/HEFT↓ | 校准评分C/HEFT↓ |
|---|---:|---:|
| GNN+LayerNorm BC | 0.91367 | 0.75993 |
| GNN+LayerNorm PPO | 0.91176 | 0.75993（BC step0） |
| GNN+LayerNorm BOPO | 0.91120 | 0.75993（BC step0） |
| GNN+LayerNorm PPO+KL1 | 0.91272 | 0.75993（BC step0） |
| Attention BC | 0.92129 | 0.87521 |
| Attention PPO | 0.92129（BC step0） | 0.76617 |
| Attention BOPO | 0.91830 | 0.77618 |
| Attention PPO+KL1 | 0.91578 | 0.76492 |
| Greedy | 1.21975 | 1.21975 |
| RandomMean5 | 1.37673 | 1.37673 |
| HEFT | 1.00000 | 1.00000 |
| PEFT | 1.06953 | 1.06953 |
| MSLBL / Portfolio | 0.65456 | 0.65456 |
| Teacher64 | 0.65112 | 0.65112 |

校准GNN BC相对HEFT逐例平均缩短24.01%，相对Greedy缩短34.21%，相对RandomMean5缩短45.10%；但相对MSLBL平均多用13.88%时间，相对Teacher64多用14.53%。原评分GNN BOPO相对MSLBL平均多用46.75%，校准缩小了差距，尚未超过强对照。

校准Attention PPO相对自身BC逐例平均改善4.83%，图聚类区间[-5.97%,16.18%]；BOPO为3.67%，PPO+KL1为4.54%。只有4个图组，且检查点按这些VAL挑选，区间不包含选择偏差。

校准GNN三个后续训练臂都选step0，不能说RL有贡献；其最终PPO相对BC退步9.62%。selected/last、逐例结果与负例全部保留。当前也没有更复杂架构必然更好的证据。

## 下一步最小闭环

1. 扩独立公开TRAIN/VAL图，先冻结清单及源模板分组；不按测试胜负筛选图，TEST继续未读取。
2. 校准GNN与容量匹配Attention，先做充分BC；去通信消息/去OCT分别作小消融。
3. 各自BC分叉PPO和一个稳定控制，保留MSLBL/portfolio/teacher参照；设置锁定后做最小多种子及独立TEST确认。
4. 论文初稿已写入实际开发结果，主实验只填实际测量，不把step0说成RL改善。

公司机器看MODULE_LAB_SERVER_PROMPT_20261019.md，不需要push。CUDA只有代码路径，尚未实测；本机CPU跑通。24例CPU冷端到端均值15.46ms，包含RPC，排除模型/数据加载，不能当A100吞吐或所有baseline时延对照。

原check_bundle本机严格PASS，106SHA全部一致；便携审计只对selected_log_probability允许1e-12漂移，并保留严格检查器实际输出。公司环境之前的数字及报错来自用户转述，本轮未假装独立审查过公司机器。

## 证据与复算

evidence/module_lab_20261019保存七阶段summary、metadata、curves、全部配置键/数据SHA、机制检查、KL门槛及重复张量校验，不上传权重。aggregate.json可用scripts/summarize_module_lab.py在保留的runs上重算，summary逐例makespan可独立复算汇总。新增文件清单MODULE_LAB_MANIFEST_20261019.json与旧BUNDLE_MANIFEST.json分别保护增量和原106文件。
