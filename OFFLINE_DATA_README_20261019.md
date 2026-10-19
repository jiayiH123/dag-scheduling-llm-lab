# 完整离线数据入口：WfInstances＋TPC-H

2026-10-19。用户要求公司电脑只能从本研究仓库pull，不能从其他地址下载。本增量把完整原始来源、已适配的物理实例、可直接训练的数据及代码放进普通Git；不依赖Git LFS、Release或首次运行联网。此说明取代旧交接中“公开数据只有12条、需在线下载”的当前状态描述；旧实验记录保留。

新增数据约240MB。用户后来提供的Halo论文与本包TPC-H查询stage不是同一种节点粒度；已附带可离线阅读的作者demo源码，具体适配建议见[Halo范围审阅](docs/HALO_SCOPE_REVIEW_20261019.md)。该参考源码尚不是可训练的agent工作流数据集。

## 拉取后先验证

备份公司机器的本地改动、runs和检查点，再git pull --ff-only。不要reset或clean。

```bash
python scripts/verify_offline_delivery.py
```

仅需Python标准库，验证完整离线包、训练数据、代码及原106/上一轮90个冻结文件。数据已在仓库中，无须执行fetch_full_wfinstances.py或任何--download命令。

## 包中数据

| 数据 | 原始/基础实例 | 全量任务数范围 | 分组 |
|---|---:|---:|---|
| WfInstances | 318条JSON执行/示例记录，所有351个上游文件 | 5～9805 | 214TRAIN、61VAL、41TEST、2合成示例 |
| TPC-H原始查询 | 154张查询图＝22种查询×7种输入规模 | 2～18阶段 | 作为原始组件完整保留 |
| TPC-H生成组合 | 1000张多查询DAG，组合5/10/20个完整查询 | 18～251阶段 | 800TRAIN、100VAL、100TEST |

TPC-H组合是基于真实查询阶段样本生成的工作负载，不是1000条独立真实执行trace，也不是作者50/100/150-query实验的逐字复现。官方TPC-H是数据库决策支持基准，本包是PPO-BiHyb作者查询DAG和新增调度场景，不包含DBGEN数据库表，不声称TPC认证结果。

每张基础图配3/6pool×两种网络×两档预算，共11776个稀疏物理配置。它们不等于11776张独立图。全量任务、依赖、文件流量均保留，没有为控制大小截断大图。

数据位置：

- data/offline_20261019/INDEX.json：全部基础图、分组、尺寸、稀疏记录路径。
- adapted/目录：每图8配置的gzip JSONL，含预算/资源/通信参数，尚无教师标签。
- base_catalog.jsonl.gz：原始组件关系、任务信息和组合来源。
- raw/目录：完整原始来源压缩分片，单文件最多32MiB，普通Git直接传输。
- SOURCES.json：上游版本、逐文件Git对象/SHA256及来源说明；原许可和README保留。
- data/offline_ready_20261019/：已经物化为当前训练器格式的TRAIN/VAL及TRAIN教师标签，见下一节。

原始完整文件可完全离线解包：

```bash
python scripts/verify_offline_bundle.py --extract-sources runs/offline_sources_new
```

输出目录必须不存在。脚本验证压缩分片、重建归档、安全解包并复核973个上游文件。所有源文件实际包含在Git中，不是仅有下载链接。

## 直接训练的数据

全量稀疏库覆盖所有尺寸。当前GNN/Attention的数据接口使用稠密data_mb矩阵，因此另外预物化不超过512任务的TRAIN/VAL；这是明确的规模子集，大图仍在全量库中。准备范围按尺寸决定，没有按模型胜负筛选。

| 可直接读取的文件 | 训练底图/组合 | 训练配置 | 验证底图/组合 | 验证配置 |
|---|---:|---:|---:|---:|
| wfinstances_trainval.jsonl.gz | 123 | 246 | 35 | 70 |
| tpch_composite_trainval.jsonl.gz | 800 | 1600 | 100 | 200 |

这两个文件使用3pool、10/40MBps网络和预算eta=.25/.6。物化时已生成TRAIN的Teacher64方案及动作；VAL只含HEFT参照，不生成训练标签；TEST没有混入这两个文件。标签是启发式组合加64次局部提案，不是求解器最优解。教师动作数与数据SHA在各自.manifest.json中。

对应配置：configs/offline_wfinstances_trainval.json和offline_tpch_composite_trainval.json。train_limit/val_limit根据实际数据生成，不沿用旧64/32。它们是起始训练配方，正式运行前在公司机器本地提交训练协议，无须push。现有训练器可直接读取，例如：

```bash
python scripts/run_module_lab.py --config configs/offline_wfinstances_trainval.json --data data/offline_ready_20261019/wfinstances_trainval.jsonl.gz --device cuda --output runs/offline_wf_run_new
python scripts/run_module_lab.py --config configs/offline_tpch_composite_trainval.json --data data/offline_ready_20261019/tpch_composite_trainval.jsonl.gz --device cuda --output runs/offline_tpch_run_new
```

先检查实际GPU和依赖环境。新数据工具不需下载依赖；训练仍使用公司已有Torch、NumPy、OR-Tools等环境。若分属两个环境，加--sim-python指定模拟器解释器。CUDA尚未在本机验证，不能把本机CPU工程检查称为服务器吞吐验证。

两条数据线分别跑、分别报告。可依据开发结果选择论文主问题与模型设置；最终测试固定所选范围并报告完整结果，不能只保留获胜实例。

## 如何离线扩展物化范围

原始和已适配的全部数据均已具备；更换pool、网络、预算或规模不需要外网。选择更大尺寸前先评估当前稠密矩阵及Attention显存开销，不能把数据已齐全等同于当前训练器已具备9805任务扩展性能。

```bash
python scripts/prepare_offline_training.py --dataset wfinstances --pools 3,6 --profiles 10_40_MBps,100_400_MBps --max-tasks 512 --output runs/wf_more_profiles_new.jsonl.gz --config-output runs/wf_more_profiles_new.config.json
python scripts/prepare_offline_training.py --dataset tpch_composite --pools 3,6 --profiles 10_40_MBps,100_400_MBps --output runs/tpch_more_profiles_new.jsonl.gz --config-output runs/tpch_more_profiles_new.config.json
```

默认为所有符合范围的TRAIN/VAL生成教师/参照；可用--train-bases和--val-bases限制工程试验。输出拒绝覆盖。TEST稀疏物理记录已提供，最终评估需另锁协议；本次没有测试任何训练后模型。

## 约束与假设

- 固定依赖DAG、不可抢占；跨pool传输按数据量/带宽算延迟，可重叠、无链路争用，同pool延迟0。
- pool累计容量约束：WfInstances用需求1/容量2的模拟槽位；TPC-H用作者stage任务数量作需求、每pool容量600。
- 相对算力[1,2,4]，费用率[1,4,16]；6pool时重复该三类。它们是模拟平台，不是实测所有设备执行时间。
- 硬预算为总执行费用上限B=Bmin+floor(eta×(Bfast−Bmin))，eta=.25/.6，不包含网络计费或空闲租费。
- WfInstances任务runtime和边共享文件大小来自原始记录；runtime向上取整至4ms，保留完整原始值。
- TPC-H按作者清理first-wave样本的方式聚合时长，按毫秒解释用于新增网络模型；通信量缺失，使用固定种子的模拟边数据量。该假设必须在论文中披露。
- WfInstances按有向拓扑指纹分组并保留旧开发来源；TPC-H组合按查询类型分组，组合指纹跨TRAIN/VAL/TEST无重合。单个原始查询的常见拓扑可跨类型重复，不作为严格结构泛化主实验。

本次按用户要求下载了全部TEST原始数据，替代此前“不下载保留路径”的传输策略；没有用TEST生成训练标签、调参或评价模型。数据准备和最低费用串行可行见证不是性能收益。

完整构建、离线解包、旧96配置物理等价及训练接口检查见evidence/offline_*_20261019.json。没有新增PPO/LLM训练结果。后续科研记录遵循autoresearch；ARA只在任务结束后归档。
