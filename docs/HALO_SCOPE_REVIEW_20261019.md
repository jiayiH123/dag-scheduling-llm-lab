# Halo 与当前课题：数据粒度和适配边界

2026-10-19；用户提供 https://arxiv.org/pdf/2509.02121v2 。本次是论文/代码审阅，没有运行Halo或生成其执行trace。建议仍属AI建议，未获用户确认转题。

论文§6.1/Table3：作者手工设计六类LLM与工具混合工作流；其中W5/W6使用PostgreSQL中的TPC-H SF10表及参数化SQL。Table3列W5为7个GPU节点/9个CPU节点，W6为9/12；1024是批输入请求数，不是单图节点数。正式目标§3使用epoch时延/总负载及开销，未写入我们的货币执行费用硬上限。调度用DP，不是神经网络单forward。实机用3张H200 NVL。

来源：[arXiv v2](https://arxiv.org/pdf/2509.02121v2)，[作者代码](https://github.com/mlsys-io/Halo_demo)，[实验模板说明](https://github.com/mlsys-io/Halo_demo/blob/main/templates/exps/README.md)。

## 三种TPC-H不能混称

| 数据线 | 节点含义 | 当前状态 |
|---|---|---|
| 官方TPC-H | 数据库表、SQL及评测规范；自身不是agent DAG数据集 | 本包不含DBGEN数据库 |
| 本包PPO-BiHyb-derived | SQL查询内部的执行stage；组合多个查询共同争用pool | 154原始查询、1000生成组合及ready TRAIN/VAL已提供 |
| Halo | LLM调用、完整SQL调用、HTTP/本地工具调用 | 已离线附带作者demo源码/模板；未采集执行时长、输出大小、设备性能或生成训练数据 |

可从Halo借用应用背景和模板结构；不能把现有查询stage改名为agent节点，也不能把公开模板等同于完整实测训练集。

## 与我们当前环境的关系（本项目分析）

共同点是DAG依赖、算力/资源差异与端到端完成时间。主要区别如下。

- 我们的runtime(task,pool)和edge_bytes目前是固定输入；Halo的模型驻留、前缀/KV复用会使实际执行时长依赖历史状态。当前模拟器不表达这些效果。仅借用拓扑时，应称Halo-derived固定成本调度场景，不声称复现完整LLM运行系统。
- CPU SQL和GPU LLM需要执行资格掩码、不同资源维度与可解释的设备计费。现有3/6个抽象pool不是自动成立的CPU/GPU部署，Wf需求1/容量2也是模拟假设。
- LLM输入输出、SQL结果和KV迁移是不同通信对象。边的payload大小、传输链路、缓存状态应分别测量或披露模拟；不能给所有跨pool边随意套同一个实测标签。
- 同一个请求预算与整个批次总预算需要先选定。保持任务逻辑和输出质量不变时，设备分配的执行费可用我们的硬约束；更换模型/缩短输出则引入质量约束，应另立问题。

因此不建议因新论文立即推翻现有训练代码。优先用已齐备的TPC-H组合和小中型WfInstances验证策略/训练目标，再决定是否增加agent应用层。若主张解决agent系统特有问题，需增加资格掩码、测量数据及相应系统基线，不能只用Greedy/HEFT就作完整系统优越性结论。

## WfInstances的9805不是实验起点

完整保留所有原始文件是传输完整性要求，和论文主实验规模是两回事。本次按INDEX元数据统计，去掉2个hello-world示例：<=50任务29条、51–100任务27条、101–200任务47条、201–512任务77条；总计180条<=512，其中103条<=200。未根据策略表现筛选。

现成ready文件含123 TRAIN/35 VAL底图、246/70配置，范围11–507任务。若主线限定中小型工作流，可在看结果前按完整图大小另锁例如20–200的范围；所有分组沿用拓扑隔离，不裁断大图凑尺寸。9805任务只作为可选规模扩展，不是本文必须达标的指标。

科学工作流与agent工作流语义不同，不能仅因节点数相近就宣称相同应用；但若论文定位为通用异构DAG调度，小中型科学工作流可以作为独立应用验证。

## 下一步最小适配方案（未执行）

1. 从W5/W6静态模板开始，保留全部依赖。先只测现有本地模型和SQL/工具的任务时长及输出字节；按模型、输入长度、设备记录，训练时重放这些统计，无须每个RL episode都实际调用LLM。
2. 使用固定模型和输出参数，明确CPU/GPU资格、容量及按时长计费；先关闭缓存复用，确保HEFT、portfolio、Teacher64和学生在完全相同的固定成本环境下比较。
3. 按模板/请求源分组TRAIN/VAL/TEST，防止复制同一模板几千次后把请求数误当独立拓扑数。只有两个TPC-H模板，不能支持广泛未见结构泛化结论。
4. 若固定成本层已有增益，再做少量实机端到端验证；缓存/模型切换另作扩展，不混进原有结果。

## 附带的离线代码及版本差异

`data/halo_reference_20261019/source.tar.gz`是49个文件的作者demo源码归档，GitHub归档pax头给出commit `ebba1634cdc15067b9a73a190c213546d74fb74f`，逐文件SHA见SOURCES.json，Apache许可原样保留。校验后可在新目录解包阅读；不必安装或执行其中代码。

```bash
mkdir -p runs/halo_reference_readonly
tar -xzf data/halo_reference_20261019/source.tar.gz -C runs/halo_reference_readonly
```

当前demo的模板README将W5/W6列为7/10和9/13，比论文Table3各多计一个rule processor，并注明HTTP实现与论文实验不同。应固定此commit、重新按parser统计节点，不把它当作v2逐字复现。demo不提供数据库加载器或完整实测trace；本次没有下载模型、TPC-H SF10表、IMDb或FineWiki。公司电脑仍可离线阅读全部代码和模板，但尚不能据此直接运行完整Halo评测。
