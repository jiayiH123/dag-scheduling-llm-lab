# Qwen/GNN共用结构化调度数据 v1

用户已确认执行两条并行路线：本地架构/训练探索，公司机器尝试Qwen1.7B结构化策略。最新用户指示覆盖旧交接中“LLM路线延后”的优先级描述。本增量构造数据和标签，尚未下载、实现或训练Qwen策略。

## 公司电脑入口

备份本地代码改动、runs和检查点，再拉取本研究仓库；没有外部数据、模型Hub或Git LFS下载依赖。

```bash
git pull --ff-only
python scripts/verify_structured_delivery.py
```

验证使用Python标准库，包括新分片/标签、真实图指纹与分组、此前完整Wf/TPC离线包、原106及上一轮90冻结文件。新说明见docs/STRUCTURED_QWEN_HANDOFF_20261019.md。

## 数据规模

| 分组 | 独立底图 | 原始记录 | 不同物理配置 | 去重后训练标签 |
|---|---:|---:|---:|---|
| TRAIN | 10000 | 20000 | 17261 | 1122860个动作 |
| VAL | 500 | 1000 | 862 | 无；只有HEFT参照 |
| TEST | 1000 | 2000 | 1718 | 无 |
| OOD-size | 400 | 800 | 694 | 无 |
| OOD-pools | 400 | 800 | 711 | 无 |

底图是一张固定依赖/工作量/需求的DAG。每图先抽一套平台/网络，再保存两个预算条件版本；不同底图使用不同抽样参数。预算对从η=(0.1,0.5)、(0.25,0.75)、(1,nonbinding)中选取：Bη=Bmin+floor[η(Bfast−Bmin)]，Bmin是各任务最便宜pool费用之和，Bfast是各任务最快pool费用之和，nonbinding的Bmax是各任务最贵pool费用之和。

**数据质量纠正：24600是原始记录数，不是24600个不同物理问题。** 对3354张底图，Bfast=Bmax，η=1和nonbinding实际预算相同，所以共有21246个不同物理配置。原始TRAIN的20000记录/1300000标签保留为审计归档；训练与评估默认使用17261条/1122860标签的去重视图。它固定保留相同(base_id,实际预算)的第一条记录，不比较教师成绩。重复记录的教师局部搜索种子不同，其中2573对TRAIN映射不同，不能当成额外独立问题。各模型和baseline使用相同去重视图；主数据和原协议不改写。实际审计见evidence/structured_v1_unique_view_audit.json。

主分布N20/40/80/120、P3/4/6；OOD-size只换成160/200任务，OOD-pools只换成8pool。全局有向WL16指纹分组/去重，所有配置都在所属底图分组。指纹核验针对实际序列化图；底图、配置、动作数量分别统计。

这是全新合理simulation，不是真实生产trace，也没有把TPC-H阶段或Wf任务改名为agent。五类结构、平台速度/价格/容量、通信与预算独立变化；没有按教师或学生胜负筛样。现成公开工作流包保留，供独立应用验证。数据清单、实际参数计数和检查结果以data/structured_scheduling_v1/MANIFEST.json及evidence/structured_v1_full_*为准。

原始20000条TRAIN记录上，教师相对HEFT逐实例平均缩短15.7487%，相对MSLBL适配缩短9.9652%；这些是原始标签程序的质量，不是去重视图统计，未运行Qwen或新模型训练，也不是OPT空间估计。所有普通/负例均保留。

## 样本格式和读取

分片在data/structured_scheduling_v1/{train,val,test,ood_size,ood_pools}/。每个gzip JSONL分片最多250条，每条是一份完整DAG及平台，不裁切依赖。BASES.jsonl.gz是所有底图种子/分组；CONFIG.json保存冻结参数。

```python
from daglab.structured_dataset import numeric_view
from daglab.structured_unique import iter_unique_rows
row = next(iter_unique_rows("data/structured_scheduling_v1", split="train"))
x = row["input"]
view = numeric_view(x)
print(row["key"], view["output_shape"], view["position_count"])
```

- input：任务workload/需求、逐pool时长/费用、边和MB、pool速度/容量/费率、带宽、预算、HEFT顺序、归一化尺度。
- supervision：仅TRAIN有教师映射、每步真实可行pool/EFT/预算历史、动作、start/end、makespan/费用。GT是设备动作，不是a/w参数。
- quality/meta：来源、参数、教师规则和搜索工作等审计信息；不要送入模型。模型输入API只接受input。

iter_rows仍读取原始归档；iter_unique_rows用于训练/模型评估，limit在去重之后计数。底图分组保持原样，不发生TRAIN/VAL/TEST迁移。原始130万动作=四档任务数20/40/80/120各5000条配置对应的任务数之和，不是130万个DAG。

## Qwen结构化输入/输出

建议使用公司已有Qwen3-1.7B权重；实际Base/后训练版本、目录和revision需记录。本机没有下载模型。Qwen作为可微骨干：完整数值前缀→所有已知task-pool查询→一次完整forward→共享head输出[N,P,2]的a/w→NormalizedEFT/LOCAL解析调度。

numeric_view每种记录一位置，包括global/task/pool/edge/link/assignment。**全部task-pool执行时长/费用assignment记录都在前缀中，所有query都在完整前缀后面**，满足因果模型的信息可见性。类型专用小投影层和共享ID编码由策略实现；不能仅把早期任务token的隐藏向量当作已看到后面所有任务。

```python
from daglab.structured_torch import numeric_batch, bc_loss
batch = numeric_batch([row], device="cpu")  # numeric_features[B,L,6]，另有类型/ID/attention_mask/query_positions
# Qwen策略需自行实现：从query hidden states输出有界pair_coefficients。
# loss = bc_loss(pair_coefficients, row["supervision"])
```

这是动作监督SFT/BC格式，需自定义policy/head训练；不是直接交给聊天SFT工具的assistant文本数据。a/w经8*tanh(raw/8)有界，动作分数=a+w*budget_local+relative_EFT；bc_loss在教师自己的可行集/前缀上求负log-prob。后续PPO必须重新采样当前策略轨迹并重放自身前缀。任何文本generate、解释、反思或多候选测试选优均另计forward/推理工作。

numeric_view的位置数L=1+N+P+E+P²+2NP：全局1、任务N、pool P、边E、网络P²、输入task-pool特征NP和输出查询NP。实测最长3219位置，限制在4096以内；一位置可含多个数值，不是一个任务或一个动作标签。这不等于把JSON原文交给Qwen tokenizer也只有4096token。数值ID、类型和feature需要可训练投影/编码，不能直接当成词表tokenID。Qwen策略、LoRA、CUDA性能和训练后效果仍待服务器实现/验证。

## GNN/Attention兼容桥接

母数据保持稀疏，Qwen训练可流式读取去重视图。现有module_lab使用稠密图并缓存记录，建议先显式选定规模物化，不默认把全部17261条塞进旧缓存：

```bash
python scripts/materialize_structured_training.py --data data/structured_scheduling_v1 --output runs/structured_module_new.jsonl.gz --train-limit 1000 --val-limit 100 --config-output runs/structured_module_new.config.json
```

该命令只读取TRAIN/VAL，拒绝覆盖已有输出；训练配置标注DRAFT，正式运行先本地提交训练协议。Qwen/GNN/Attention若比较骨干，使用相同底图、配置、教师、物理信息和固定HEFT+normalized解码。新输入派生特征应给各模型同等使用。

## 教师和物理语义

教师是Teacher64程序：HEFT、PEFT、JoinLookahead、CostFirst、MSLBL适配及price-PEFT(0.1/0.3/1/3/10)，共十规则，提取映射后统一固定HEFT顺序重建，先选最好的可行映射，再64次一/二任务局部提案。预算内优先makespan，费用作平手处理；64是尝试次数，不是64次成功改善。与原教师逐例一致性已检查；没有求解器最优或神经教师声明。搜索所得质量不是部署策略结果。监督GT是每个任务最终选择的pool，不是教师神经参数，也不是人工a/w数值。

固定延迟，可重叠、无链路争用，同pool通信0；单维累计容量、不可抢占。时长为参考秒/相对速度后向上取整成毫秒tick；费用=时长tick×模拟整数费率。网络MBps、资源、价格、任务量均为模拟。预算使用Bmin/Bfast分档；真正nonbinding用逐任务最大费之和。数据没有加入KV缓存/模型切换/LLM质量约束，因此不声称复现Halo系统。

## 复现和检查

协议ee4b4a8先于任何数据生成，生成实现46e1b7a；source_hashes约束实际生成文件。所有数据确定性生成，运行时长只在manifest，不在样本；8个TRAIN记录独立再生成完全相同。pilot先1000底图/2000配置，通过后保留为主TRAIN子集，不重复计数。

```bash
python scripts/check_structured_semantics.py --data data/structured_scheduling_v1 --output runs/structured_label_audit_new.json
```

该检查独立事件扫描验证序列化TRAIN调度、通信、费用与每步预算/概率。原始数据构建需模拟器的NumPy/OR-Tools环境，但交付的实际数据读取/静态校验不需要它们。没有新神经优化、Qwen加载、CUDA或训练后模型TEST结果。
