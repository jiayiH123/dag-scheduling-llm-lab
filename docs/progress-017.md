# LLM能做什么：角色、数据和单次forward

2026-10-15；外循环36，文献/代码核对与方案筛选。没有训练、下载模型、远程连接或LLM调用。全部收益预期仍未验证。

## 判断

可以小规模试，但没有证据LLM天然比GNN适合当前纯数值DAG。只换backbone不会扩大固定HEFT顺序的可达调度，也不会自动修复PPO的信用分配。语言先验最直接的用途是离线生成更好的规则/搜索邻域；如果让LLM参与调度，更合适的一次forward角色是全图结构化评分与任务优先级，而非生成开始/结束时间。

## 三种角色

| 角色 | 输入 | 输出 | 推理边界 |
|---|---|---|---|
| 离线规则设计器 | 模型、规则接口、训练/选择实例的失败统计 | 新的打分函数或邻域代码 | 部署运行冻结代码，不调用LLM；生成规则要单独作强baseline |
| 全图决策编码器 | DAG、任务/资源池属性、通信矩阵与预算 | 每任务优先级、每任务—池评分、预算升级价值 | 完整实例编码一次；程序执行硬约束、时序构造 |
| 预算/通信宏观控制器 | 完整实例及分支/关键路径摘要 | 升级价值或代价权重 | 软建议，不能替代精确全局预算检查；需与小MLP/GNN控制器比较 |

只选择现有十规则中的一个不会超过同候选、同解码集合的逐实例最优值；不适合作为突破该集合的主机制。逐任务反复问LLM和完整文本调度均不满足严格单次forward方案。

## 输入输出与数据

数据物理实例可以复用，需要新的表示适配：任务workload/demand、带数据量的稀疏边表、资源池speed/capacity/price、带宽矩阵、精确预算和费用/通信语义。可变长度紧凑JSON或数值embedding均可，精确数值保留给物理解码器，所有模型获得同等信息，底层图划分保持隔离。

严格一次forward可设计为“完整输入前缀＋所有任务/任务池查询位置→LLM隐藏向量→共享评分头”。不能调用generate输出一长串JSON后声称只有一次forward；自回归生成通常反复调用模型。decoder-only模型的早期任务位置看不到后续输入，后置查询必须放在完整实例后；这是待实现验证的设计。

先只输出与当前GNN一致的a,w评分，保留同一解码器，隔离backbone差别。之后再加入任务优先级、扩大动作空间，并同步给GNN对照。程序负责预算、资源日历、依赖/通信与时刻，奖励来自真实模拟makespan。

现有800训练配置36540动作标签可用于新输入表示的BC。学习更好的任务顺序需要额外的多顺序专家/联合搜索示范，或由RL学习；不能把现有固定HEFT顺序标签说成更好顺序监督。这里用动作交叉熵和PPO，不必转成聊天回答的token损失。初始冻结backbone只训head，再决定是否LoRA；实际modelID和服务器尚未确定。

## 最小验证顺序

先确认固定顺序映射空间与现有训练是否充分。LLM小实验保持相同信息/标签/解码器，与GNN和小Transformer比较，各有BC与BC+PPO、至少3种子。若LLM有稳定信号，再扩规模和改变顺序。评价保留适配预算baseline、教师质量时间参照、总参数/可训练参数、显存、完整推理耗时与forward计数。不存在“换LLM就能提高指标”的结论。

来源：[ReEvo代码](https://github.com/ai4co/reevo)、[EoH ICML2024](https://proceedings.mlr.press/v235/liu24bs.html)、[LLMs can Schedule作者论文](https://www.lamsade.dauphine.fr/~cazenave/papers/LLMsCanSchedule-DSO-2025.pdf)、[Qwen接口](https://huggingface.co/docs/transformers/model_doc/qwen2)、[自回归生成](https://huggingface.co/docs/transformers/v4.50.0/en/cache_explanation)。前述工作支持可借鉴的角色/接口，不是完全相同场景的效果证明。

完整[候选与证据](../literature/llm_roles_20261015.md)、[未执行的试验草案](../experiments/llm_role_pilot/protocol.md)。
