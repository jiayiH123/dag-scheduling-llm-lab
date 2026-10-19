# 模块筛选入口与公司电脑继续实验说明

本目录是增量交付，原106个冻结文件不改；不覆盖服务器已有的Attention/BOPO代码、runs/或检查点。
日期2026-10-19。实际结果及全部负例见`evidence/module_lab_20261019/`和本次结果报告。

## 1. 本次模块究竟是什么

| 开关/名称 | 用途 | 必须保留的对照 |
|---|---|---|
| legacy_gnn | 旧12维特征＋加权双向DAG消息 | 新实现的共享双输出head、12866参数；不是冻结14946参数原GNN的逐字复现，也未使用历史warmstart |
| physical_gnn | 追加OCT、深度、候选时长、入/出通信统计等8维 | 与legacy输入作信息增量比较，不说纯网络框架贡献 |
| edge_gnn | 显式传入归一化链路延迟，不只用1/(1+delay)衰减邻居 | 与physical_gnn共享20维输入；旧边特征失败试验不自动等同本设计 |
| edge_gnn_ln | 上述GNN每层后加LayerNorm | 13314→13442参数，隔离归一化作用；目前是有价值的简单对照 |
| edge_attention | 显式DAG消息＋任务轴Attention＋pool轴Attention | 30466参数；比GNn大，需要后续容量匹配，不能把全部差异归给Attention |
| physical_mlp | 不传DAG邻居消息的简单对照 | 同20维输入；避免只比较复杂方法 |
| edge_gnn_oct | 把OCT差作为有界静态评分先验 | BoundedOCTPriorOnly独立控制，不能把解析先验冒充学习收益 |

所有策略输出每个(task,pool)的a,w，固定HEFT构造顺序、cap8。前几轮沿用LOCAL的`a+w*phi−EFT/scale`；新增公开校准轮仅把当前可行pool的EFT先验平移/缩放到[-1,0]。原vendor不改，新decoder隔离实现。SGS仍可并行/插入，并严格检查依赖、通信、累计容量和硬预算。没有新增task排序/skip，也没有部署时多方案取最优。

GNN主要沿上下游边逐层汇总；我们的Attention在这些消息上再让任务和pool互相选择关注对象。Attention不是LLM，没有语言模型权重、文本生成或自回归调用。两者均一次完整forward，随后一次确定性构造。

算法来源为借鉴而非原论文复现：

- [GiPH/MLSys2023](https://proceedings.mlsys.org/paper_files/paper/2023/file/3e3eec95971350490e37a076fdc100ad-Paper-mlsys2023.pdf)：product graph与通信消息的相关思路。
- [ReSched/ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/467e109270678b3e2b3c2e7fa6b3d575-Abstract-Conference.html)：任务/机器交互Attention；本实现是静态DAG/pool适配，不照搬其动态FJSP策略。
- [PEFT/TPDS2014](https://ieeexplore.ieee.org/document/6471969/)：OCT后继代价；沿用仓库已有适配公式。
- [BOPO/ICML2025](https://proceedings.mlr.press/v267/liao25a.html)：最佳完整候选与其他候选配对，`softplus((C_loser/C_winner)*(meanlogp_loser−meanlogp_winner))`。普通pairwise去掉费用比，SIL只模仿优于当前greedy的最佳候选。

PPO是actor-only clipped-policy训练，使用自greedy全局回报baseline，无critic/GAE。`ppo_kl1/ppo_kl10`额外约束当前策略不要过度偏离冻结BC，属于训练正则；其参考网络forward只发生在训练，部署没有第二策略或回退搜索。

## 2. 安全拉取与检查

先查看`git status --short`，对公司机器新增代码/结果做好备份，再`git pull --ff-only`。不要执行reset、clean或覆盖服务器目录。若新文件同名冲突，先移动本地副本，不删除。

```bash
python scripts/verify_module_lab_addon.py
python scripts/audit_bundle_portable.py --output runs/module_bundle_audit_new.json
```

第一条只验证原106文件及新增文件SHA，不需要Torch。第二条保留原`check_bundle.py`的实际PASS/FAIL，只允许`selected_log_probability`在1e-12绝对/相对容差内漂移；动作、mask、预算、offset、教师调度及SHA仍严格检查。不是把原严格检查的失败改写成PASS。

当前环境只在CPU验证过。GPU需先使用现有preflight，再运行新增检查入口：

```bash
python scripts/server_preflight.py --require-cuda --output runs/module_gpu_preflight_new.json
python scripts/run_module_lab.py --device cuda --checks-only --output runs/module_gpu_checks_new
```

若Torch与模拟器在不同环境，给`run_module_lab.py`添加`--sim-python /path/to/simulation/python`。本机正是Torch与NumPy/OR-Tools分开的两个环境；新接口通过JSON通信，不要求重装系统环境。

## 3. 可执行的筛选命令

```bash
python scripts/run_module_lab.py --config configs/module_lab_local.json --device cuda --output runs/module_screen_new
python scripts/run_module_lab.py --config configs/module_lab_followup.json --device cuda --output runs/module_followup_new
python scripts/run_module_lab.py --config configs/module_lab_replication.json --device cuda --output runs/module_attention_replication_new
python scripts/run_module_lab.py --config configs/module_lab_gnn_ln_replication.json --device cuda --output runs/module_gnn_replication_new
python scripts/run_module_lab.py --config configs/module_lab_bc_kl.json --device cuda --output runs/module_kl_new
```

这些是验证/重复本机已完成开发实验的命令，**不要默认全部重新跑一遍当下一轮主实验**。当前默认数据仍是原800TRAIN/216VAL，代码初始化即排除TEST使用。输出目录须不存在，脚本拒绝覆盖；若中断，已完成臂的JSON/检查点和曲线仍保留，新一轮使用新目录。

首轮先6个BC选前2个架构，再分叉4种目标；后续归一化控制/复验的协议均先独立commit。B是真实完整调度数量（含1greedy）；PPO只用随机候选做on-policy比率，greedy作baseline。每条候选自己的可行集/EFT/预算历史用于概率重放。K仅用于偏好保留，不等于真实采样量。不同目标优化epoch数不同，实际候选、梯度步和耗时分别记录。

## 4. 下一步推荐，避免重复已完成工作

**新增可执行公开数据入口**：已固定8TRAIN/4VAL新执行trace、22–233任务，3/6pool×2网络×2预算；3条TEST路径只预留，不下载。原6条公开smoke不在这份清单中。它是小型公开开发实验，不是最终主测试。

```bash
python scripts/prepare_module_public.py --download --raw-dir runs/module_public_raw_new --output runs/module_public_development_new.jsonl.gz
python scripts/run_module_lab.py --config configs/module_lab_public_pilot.json --data runs/module_public_development_new.jsonl.gz --device cuda --output runs/module_public_pilot_new
python scripts/run_module_lab.py --config configs/module_lab_public_scaled.json --data runs/module_public_development_new.jsonl.gz --device cuda --output runs/module_public_scaled_new
```

先核对原JSON的Git blob/SHA与TRAIN/VAL拓扑指纹，再仅给TRAIN生成教师标签；VAL教师只作评价参照。网络和预算仍是声明的模拟参数。float32网络配float64条件概率，用于较大实测时长范围的数值一致性。生成包的gzip头和计时字段可能因机器不同而变化；data_sha用于本次追踪，不承诺跨机器压缩包逐字相同，原始source校验仍严格。

1. 先读`MODULE_RESULTS_20261019.md`。公开校准轮的GNN BC为0.75993C/HEFT，Attention PPO为0.76617，仍落后MSLBL的0.65456。此轮只有8张TRAIN/4张VAL底图。优先扩独立公开底图，勿在这32VAL上无限调参。最好运行可以单列，已有多种子均值同时保留。
2. 按原`SERVER_EXPERIMENT_PLAN_20261018.md`冻结**新的公开TRAIN/VAL/TEST底图清单**，再扩预算/通信/资源变体。历史6trace烟测不作为新TEST。新公共TRAIN才生成教师动作；TEST教师只能作质量参照。
3. 用相同物理信息比较GNN+LayerNorm与Attention，并增加容量匹配。当前hidden32的Attention约3万参数，不能与1.34万参数GNN作纯架构归因。
4. 确定BC已学好后再分叉RL/偏好，保留独立BC、MSLBL适配、十规则portfolio和Teacher64。原框架简单规则赢了就如实记录，不能只保留HEFT。
5. 新训练可复用`daglab/module_lab`的表示、device-aware概率和目标函数；先在新公开开发fold运行，锁设置后另做独立TEST执行器/协议。当前入口刻意不评估TEST，不能把VAL改名为新TEST。

完整新数据须满足现有record schema：instance包含物理参数，graph为旧graph_features结果；新增20维和显式边信息在RPC中从instance重算。TRAIN有teacher_decisions，VAL无训练标签。公开增强参数及同底图分组保持可审计。

## 5. 输出与只pull机器如何交回结果

每轮有metadata、完整选中键/数据SHA、curves、selected/last检查点、逐例C、候选诊断、强baseline、区间与失败记录。训练best-of-B不作部署成绩。区间按latent图聚类，属于已知VAL探索性区间，未包含检查点/模块筛选偏差。

```bash
python scripts/export_module_lab.py --run runs/module_followup_new --output exports/module_followup_for_copy.json
```

不需要push。可复制该小JSON回来，保留服务器原summary/curve/selection/checkpoint。若要端到端时延，再用：

```bash
python scripts/benchmark_module_lab.py --checkpoint runs/module_followup_new/edge_attention_s11_ppo.pt --device cuda --output runs/module_followup_new/cold_timing.json
```

时延包含每例冷特征/decoder准备、张量转换、模型、输出拷贝、JSON RPC和一次构造；排除模型/数据加载和一条TRAIN暖身。CPU本机测量不能当A100实测吞吐；CUDA路径须实际验证。不自动使用torch.compile或bf16，先保持概率重放正确。
