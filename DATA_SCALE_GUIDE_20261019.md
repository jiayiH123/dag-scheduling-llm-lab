# 仓库里有什么数据，下一轮该用多少数据

日期：2026-10-19。核对结果见[evidence/data_scale_audit_20261019.json](evidence/data_scale_audit_20261019.json)。这是数据盘点与后续计划，没有新增训练或扩充完成的声明。

## 当前实际内容

| 数据 | Git中是否有实际数据文件 | 训练 | 验证 | 测试 |
|---|---|---|---|---|
| 原合成参考集 data/reference_dataset.jsonl.gz | 有，包含TRAIN教师动作 | 800实例/800底图ID | 216配置/24底图ID | 432配置/48底图ID，已见 |
| 后续known_cycle24_test.jsonl.gz | 有，无训练教师标签 | — | — | 432配置/48底图ID，已见 |
| 新公开WfInstances pilot | 原始/转换后数据未提交；提交下载清单、准备脚本和实验记录 | 8执行trace→64配置 | 4执行trace→32配置 | 3路径仅预留，未下载 |

原合成TRAIN仍有36540个教师动作标签，20/40/80任务、3pool。没有把原来的800训练实例替换成8张公开图。表中的底图按latent_id计数，不代表800种互不相同的无权拓扑。

本轮12条公开开发trace合计2168426字节。下载后的原始JSON和准备后的数据在本机ignored runs/下；git pull不会自动运行下载。data/module_public_sources_20261019.json还含3条保留TEST元数据，默认脚本跳过它们。

## 当前服务器入口意味着什么

- 不传--data时，scripts/run_module_lab.py默认读取原合成参考集；实际使用多少由train_limit/val_limit决定。检查metadata和data_selection，不能只看文件里总共有多少行。
- configs/module_lab_public_pilot.json和module_lab_public_scaled.json配--data指向准备后的公开数据时，只使用64TRAIN/32VAL。它们的名字和分类均是pilot，不应作为正式主训练配置。
- 当前公开数据不足以判断LLM微调是否有优势，也不足以支持公开数据泛化结论。较小数据有助于调通接口，但不能靠增加同图预算变体替代独立工作流。

## 下一轮优先级：先补独立实例，再扩大模型

以下是建议规模，不是已生成的数据或已执行协议。

1. 扩充真实公共开发来源。先清点Pegasus的135条执行trace，按应用、任务规模和源模板分组；剔除重复拓扑跨split与不完整记录，保留此前已见数据标记。不能声称135条执行就是135张不同结构的图。新TEST源保持不用于训练或选择。
2. 主训练采用真实模式支持的合成数据加公共TRAIN。首阶段目标至少1000个独立生成的训练实例，后续按曲线决定是否扩到数千。可用WfCommons/WfGen，但recipe拟合只能使用TRAIN来源；生成图明确标为合成。不同结构/规模应占实质比例，不能仅复制原8图并变更费用或预算。
3. 真实公开VAL/TEST单独评价；所有同底图的预算、网络、pool变体归在同一split。若采用跨应用测试，须在协议中单列，不混成普通同分布测试。
4. 数据清单冻结后只为TRAIN生成教师动作。GNN/Attention、直接LLM、文本SFT、结构化LLM使用相同物理实例、信息和评价规则；文本与结构化教师目标由同一合法映射导出。
5. 新主实验配置必须明确数据路径、实际train_limit和选中键，避免准备了大数据却仍截取64配置。准备后报告执行trace数、生成实例数、拓扑指纹数、平台配置数和动作标签数，分别统计。

本轮只完成盘点和规模修正，尚未下载全部Pegasus、生成1000新实例或重训。公司机器可继续使用已包含的800合成TRAIN做工程验证；正式LLM比较前先执行新的公开数据扩充协议，不继续把12-trace pilot当完整训练包。

依据：[WfInstances目录](https://wfcommons.org/instances)、[WfCommons](https://github.com/wfcommons/WfCommons)。保留固定源commit、许可、清单哈希和独立分组。
