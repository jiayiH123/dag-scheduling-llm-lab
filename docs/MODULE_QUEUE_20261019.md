# 已做、当前与待做的模块组合

更新2026-10-19。用户优先级：搭建可用架构、直接转公开数据、并行写论文初稿，避免无限扩大旧合成数据的测试。

## 当前已经实现/试验

| 层次 | 选项 | 状态 |
|---|---|---|
| 输入 | 旧12维；20维物理/OCT/通信统计；显式边延迟 | 已实现并在旧合成开发集筛选 |
| 骨干 | MLP；GNN；GNN+LayerNorm；DAG消息+任务/pool轴Attention | 已筛选；归一化GNN和Attention做过3开发种子 |
| 解析先验 | 原EFT；有界OCT评分先验 | 已测；OCT-only控制独立保留 |
| 完整解训练 | BC；PPO；最佳候选自模仿；普通pairwise；BOPO适配 | 已测；BOPO没有稳定超过PPO |
| 策略约束 | 无额外约束；到冻结BC的KL系数1/10 | 已测；温和约束降低尾部退步，但平均C/HEFT略有代价 |
| 评分尺度 | 原EFT先验；当前可行pool的相对EFT归一化 | 两轮公开开发实验已完成；校准明显改善BC，强baseline仍更好 |
| 物理模型 | 通信、累计容量、硬费用预算；固定HEFT构造顺序 | 保留；所有保存调度通过检查 |

这些是借鉴模块及领域适配，不是声称首创每个模块。当前Attention约30466参数、归一化GNN约13442，纯架构贡献还需容量匹配。

## 当前公开数据最小矩阵

**WfInstances新8TRAIN/4VAL执行trace**，22–233任务、3/6pool，两网络×两预算；64TRAIN/32VAL。拓扑/runtime/filebytes来自公开记录；资源平台、链路、单位需求和价格/预算是模拟假设。3条TEST路径保留未读取。

| 骨干 | BC | BC+PPO | BC+BOPO适配 | BC+PPO+温和BC-KL |
|---|---|---|---|---|
| GNN+LayerNorm | ✓ | ✓ | ✓ | ✓ |
| Attention | ✓ | ✓ | ✓ | ✓ |

原评分与归一化评分各执行seed11、8个训练臂，共16臂，一次forward＋一次确定性构造。见`MODULE_RESULTS_20261019.md`。8TRAIN指8张原始DAG，不能把64配置算成64张独立图。

## 队列：按这个顺序推进

1. **扩独立公开底图，而非继续扩大旧合成排列。** 已完成小型公开开发结果和七项强弱baseline；下一先冻结更多TRAIN/VAL路径和源拓扑分组，TEST继续不读取。校准评分在同一物理模型内保留，可先不增加模型规模。
2. **容量匹配与最小表示消融。** GNN+LayerNorm约3万参数对Attention约3万参数；Attention去掉显式通信消息；去掉OCT特征。每次只改变一项，避免无法解释收益。
3. **充分BC后的小规模训练优化。** 若BC尚未学好，先修复输入/初始化/LR或增加BC；从各自同一BC比较PPO和一个最有希望的控制。不继续无依据添加十种loss。
4. **只有固定顺序确实限制质量时，才测可学习任务优先级。** 一次forward输出priority+a/w，解码时从ready任务选择；与固定顺序单独分组。旧教师只有固定顺序分配标签，不能假装它已提供更优顺序标签。

后续候选而非立即开跑：大图层次化/分块（HSDAG相关思路）、WeCAN式skip生成策略、CAI式优先级训练。它们会改变解码/动作空间或假设，不能当当前固定HEFT分配模型的无缝小插件。LLM继续后置，不需要先下载模型。

## 哪些公开数据能直接用

| 数据 | 本项目用法 | 与论文数据的关系 |
|---|---|---|
| [WfInstances](https://github.com/wfcommons/WfInstances) | 当前已下载并转换12条完整执行trace | 与许多科学工作流论文同属Montage/Epigenomics/1000Genome等应用族；不冒充CAI/PEFT原表的同一预处理版本 |
| [Pegasus WorkflowGenerator](https://github.com/pegasus-isi/WorkflowGenerator) | 后续可生成传统科学DAG | 需要另配异构时间、资源池、网络、预算；不能把生成图叫真实运行trace |
| [WfCommons/WfGen](https://github.com/wfcommons/WfCommons) | 可按TRAIN来源recipe扩训练规模 | 基于真实模式的合成，标签须重新生成，源模板分组避免泄漏 |
| [PPO-BiHyb TPC-H](https://github.com/Thinklab-SJTU/PPO-BiHyb/tree/main/dag_data/tpch) | 次级资源packing基准 | 原格式不直接等于异构通信预算场景；需说明新增属性/转换，不能称原论文逐字复现 |

One-Shot专有编译图或没有确认完整公开输入的数据，不作为当前最快入口。公开DAG不是“所有候选设备时长/网络/价格/容量都已实测”的完整平台，补平台是一项公开可复现的建模步骤。
