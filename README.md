# DAG Scheduling Research Lab

面向 **jiayiH123 的跨机器实验交接仓库**。目标：在通信延迟、硬预算和累积资源容量约束下，以一次完整神经网络 forward＋一次调度构造最小化全局 makespan，探索能产生稳定增益的 BC/SFT＋RL。目标会议 IWQoS，当前尚无可发表结果保证。

**2026-10-18 当前路线：结构化 Attention＋完整可行调度偏好训练。先读 [AGENTS.md](AGENTS.md)、[服务器实验计划](docs/SERVER_EXPERIMENT_PLAN_20261018.md) 和 [任务提示词](docs/HANDOFF_PROMPT.md)。仓库名保留 `dag-scheduling-llm-lab` 便于同步；LLM暂后置。新 Attention/BOPO/CUDA trainer 尚待实现，现包提供CPU参考、硬件盘点与公开数据smoke准备。**

准备资源：[论文/Git固定版本](docs/EXTERNAL_SOURCES_20261018.json)、[试验配置契约](configs/attention_bopo_pilot.example.json)。配置契约不是已有新trainer的运行入口。

## 带来了什么

- `docs/problem_model.pdf`：原问题建模；`docs/HANDOFF.md`：当前结论和待验证问题。
- `vendor/`：原物理模拟器、硬预算检查、Greedy/HEFT/PEFT、MSLBL 适配、十规则组合、GNN 和单次解码器，原文件逐字复制。
- `daglab/`：去除原路径/深层导入依赖后的便携接口、精确抽取的生成器/特征/教师、JSON RPC。
- `data/reference_dataset.jsonl.gz`：原800训练＋216验证＋432测试配置，含800训练教师和36540动作；不含测试教师标签。
- `data/known_cycle24_test.jsonl.gz`：后来用于强基线和教师诊断的432配置；与上述原测试集不同，均已被查看，不是新的盲测。
- `checkpoints/`：3种子的大图BC/PPO和早期warmstart，共9个小模型。
- `scripts/`：完整性检查、统一评测、标签导出、参考BC/PPO重放、新测试集生成、外部评分解码校验。
- `scripts/server_preflight.py`：读取CPU/GPU/PyTorch信息；`scripts/prepare_public_smoke.py`：下载并校验6条固定来源的公开已知trace，转换24个smoke配置；不代替主实验数据划分。
- `evidence/`：关键实验原始汇总/CSV，保留负结果；`archive/`：原研究状态、发现、日志和关键训练代码。
- `SOURCE_MANIFEST.json` / `BUNDLE_MANIFEST.json`：来源与逐文件校验。未携带多GB重复日志、原Git历史、第三方模型权重、账号密钥。

## 另一台机器如何开始

首次使用 `git clone`，后续更新用 `git pull --ff-only`。使用HTTPS；公开时无需登录，若仍为私有则在该机器独立认证。SSH无论公开/私有都需要密钥。

```bash
git clone https://github.com/jiayiH123/dag-scheduling-llm-lab.git
cd dag-scheduling-llm-lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

另行安装与你服务器驱动匹配的 PyTorch，参考 [官方安装选择器](https://pytorch.org/get-started/locally/)。已验证参考版本为 torch2.11.0，模拟环境 NumPy2.4.6/OR-Tools9.15.6755；不要为了迁移包覆盖已有服务器CUDA环境，可新建环境。可选LLM包列在 `requirements-llm.txt`，不需要提前全部安装。

```bash
python scripts/check_bundle.py
python scripts/server_preflight.py --output runs/preflight.json
python scripts/evaluate.py --limit 8 --teacher --output runs/smoke.json
python scripts/evaluate.py --data data/known_cycle24_test.jsonl.gz --teacher --output runs/cycle24_seed11.json
python scripts/prepare_public_smoke.py --download --output runs/public_smoke.jsonl.gz
python scripts/evaluate.py --data runs/public_smoke.jsonl.gz --split val --limit 4 --teacher --output runs/public_smoke_reference.json
```

`evaluate.py` 默认使用冻结 `local_seed11.pt`。可分别指定 `--checkpoint checkpoints/bc_seed11.pt`（复核原BC报告时同时加 `--mode initial`；与LOCAL PPO做同解码器消融时用默认local）、`local_seed22.pt`、`local_seed33.pt`。所有基线受同一物理约束。输出逐实例结果、均值、胜负计数和底层图cluster bootstrap区间；计入特征、模型RPC与解码耗时。CPU参考不自动迁移CUDA；新模型实现须独立计时/计数。

若Torch和模拟器在两个环境，评测增加 `--torch-python /path/to/torch/python`；训练增加 `--sim-python /path/to/simulation/python`。命令行默认单环境，均无本机硬编码路径。

## 参考训练与新数据

```bash
python scripts/train_reference.py --seed 11 --output runs/replay_seed11
python scripts/generate_fresh.py --seed-offset 3100000 --output runs/new_test.jsonl.gz
```

训练脚本保留原200次BC、300次PPO上限及180秒PPO墙钟上限，仅重放已有配方，**不是新的算法结果，也不保证充分收敛**。生成新测试前要锁定方案/生成器/种子；示例偏移不是永久盲测保证，生成器只检查与仓库携带图不重复。所有运行拒绝覆盖已有输出。

## LLM 接入

本节为未来备选；当前先执行[Attention＋偏好训练计划](docs/SERVER_EXPERIMENT_PLAN_20261018.md)，无需LLM权重或`requirements-llm.txt`。

阅读 [LLM 接入规范](docs/LLM_INTERFACE.md) 和 [新机器任务提示词](docs/HANDOFF_PROMPT.md)。第一阶段保持固定HEFT顺序和相同解码器，LLM一次编码输出形状 `[n*k,2]` 的a/w评分；先BC再PPO。`scripts/export_llm_data.py` 给出结构化输入，只有TRAIN带动作目标。`scripts/validate_scores.py` 可验证外部评分产生的调度，但**不能替外部模型证明forward次数**。

如果服务器只有聊天API，通常不能取隐藏向量并训练这些评分头；需要本地模型权重或自定义策略端点。普通 `generate()` 输出调度JSON是自回归多次forward，不能冒充一次forward。文本API可用于离线规则设计。

## 当前结果的边界

现有模型能在若干合成测试中平均超过Greedy/Random/HEFT，但不保证逐实例超过；在已知cycle24诊断上，MSLBL/十规则/教师仍优于学生。RL额外增益未稳定复现。不要把BC收益归于RL、把不同测试集的百分比直接拼接、把FEASIBLE解写为OPT。

这是选取的便携交接包，完整原始3GB档案仍在原电脑。旧报告保留历史相对链接，选择迁移的证据在 `evidence/`，其余用来源清单定位。详见 [NOTICE](NOTICE.md)。
