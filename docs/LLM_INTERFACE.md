# LLM接入与数据契约

## 访问方式先分清

本地Qwen/Llama权重或自定义policy服务可提供hiddenstates及可训练heads。仅有OpenAI兼容chat/completions API通常不能实现本方案的隐藏向量并行head训练；可用于离线规则生成，或明确标注为多forward文本调度对照。一次HTTP请求不等于一次forward。

## 第一阶段保持同一决策空间

`model_input`：完整DAG任务/带数据量边表、池speed/capacity/price、bandwidth、B/Bmin/Bfast及精确物理语义。导出脚本生成紧凑JSON；不要把专家解、makespan或测试baseline结果混入输入。teacheraction仅TRAIN的targets字段。所有模型可获得同等物理信息，若新增criticalpath/语义特征须给对照或隔离消融。

输入建议：完整实例前缀，后接所有已知task/pool查询位置。一次base_model.forward产生全部queryhiddenstates，接共享MLP预测每对a,w。decoder-only早期token看不到后面的任务记录，查询必须在完整实例后。数值embedding适配同样需要验证，不能声称保留了天然语言优势。

输出按node-major排列，shape[n*k,2]，例如task0pool0、task0pool1...task1pool0...。对应当前模型a=8*tanh(raw_a/8),w=8*tanh(raw_w/8)。先保持这个范围、LOCAL预算项和固定HEFT顺序，隔离backbone变化。环境从评分采样/argmax任务设备动作并更新精确mask、EFT、预算reserve，不再次调用网络，不隐藏多候选选优。外部predictions JSONL每行`{"id":"test0","scores":[[a,w],...]}`可交validate_scores校验；该脚本只证明物理调度，不证明你上游只有一次forward。

## 监督与PPO

BC：当前专家动作条件log-prob交叉熵，未必需要assistant文本token loss。建议先冻结backbone训练head，再试LoRA；记录总/可训练参数、训练样本/时间与显存。PPO对真实环境makespan计算奖励，策略采样/条件log-prob必须对应实际decoder，不能对argmax后的数组直接套一个不相符的PPO概率。

当前目标是数值DAG，不能假定LLM语言预训练能迁移。至少比较GNN BC/PPO、小Transformer控制、LLM BC/PPO；LLM若赢，再做预训练/随机初始化同架构等控制，不能把参数规模收益称为语言推理收益。

## 第二阶段才改顺序

可以输出每任务priority或预算升级机会值，但这是新动作空间/新表示。GNN也必须获得相同decoder和标签。现有教师固定HEFT顺序不能教更优顺序，需TRAIN内多顺序expert/联合order+mapping搜索示范，或RL学习。预算建议是软信号，硬总预算检查始终执行。

模型调用计数必须在完整基础模型forward处instrument，推理记录CPU/GPU预处理、forward、decoder、总延迟；GPU计时同步。文本生成、CoT、多次反思或sample256不可归为一次构造。

## 当前尚未做

没有modelID/服务器资源选择，没有模型下载、训练、checkpoint或优于GNN的LLM结果。先看docs/llm_roles_20261015.md和archive/llm_role_pilot__protocol.md的草案，具体运行协议需选择实际模型后提交。
