# 公司机器Agent：Qwen结构化SFT下一步

用户已接受结构化数值Qwen路线，同时本地继续小模型模块探索。根入口STRUCTURED_DATA_README_20261019.md取代旧“LLM延后”的当前优先级。数据已提供，先备份改动/实验目录、pull并运行verify_structured_delivery.py；不下载外部数据，不覆盖冻结记录，也不要求公司push。

1. 确认现有GPU、实际Qwen权重目录/版本与依赖；本轮没有下载Qwen或提供已训练LLM。优先评估已有Qwen3-1.7B-Base，若只有后训练版则如实记录。不要因名称相近混用不同模型。
2. 流式读取data/structured_scheduling_v1的TRAIN；选择固定VAL配置用于检查点选择。TEST/OOD只有输入，先保留。第一阶段只做少量TRAIN过拟合/完整前向计数检查，再扩大BC。
3. 实现数值projector+Qwen骨干+并行a/w头。structured_torch.numeric_batch提供类型、ID、数值、mask、查询位置；它本身不是Qwen policy。使用共享/可外推的ID表示，避免把未见任务编号只映射到未训练的独立embedding。完整assignment前缀后才接全部pair-query，调用一次base_model.forward(inputs_embeds=...)，取query hidden states，head输出8*tanh(raw/8)。不调用generate，不生成CoT。
4. 固定HEFT顺序、NormalizedEFTDecoder/local预算项与硬mask。先训练projector/head，再试LoRA；记录冻结/可训练参数、训练工作和端到端延迟。监督为structured_torch.bc_loss：教师pool的条件log-prob，不是assistant token loss或a/w MSE。原始输入、quality和supervision分开。
5. 在公司本地先提交训练协议和实际硬件/模型/seed等，不需push。最初小范围对照可复用相同TRAIN/VAL切片比较GNN、Attention、Qwen；表示比较保持同物理信息/教师/解码。完整数据主实验与pilot结果分别命名，不能称20k都已训练。
6. 从各自BC检查点分叉RL；PPO数据来自当前策略自己的采样和真实prefix。候选训练可多采样，部署保持一次forward/一次确定性构造。偏好样本若另生成只用TRAIN，标签文件不能充作on-policy PPO数据。
7. 保留BC、HEFT、PEFT、MSLBL适配、portfolio、Teacher64等质量/代价参照。训练数据教师质量不是学生性能；现有Transformer/大模型是否优于GNN，以及RL增量，都尚未验证。
8. 输出小型JSON：实际数据base/config/action计数、模型/可训练参数、BC与RL的VAL逐例结果、forward计数、时延/显存和异常；用户可复制回来，无需公司push。继续autoresearch，ARA仅任务结束后记录。

公开Wf/TPC全量离线包继续作为额外应用验证；本批主训练完全模拟。全部费用、资源和通信单位以CONFIG/physics为准；没有多维显存、缓存收益、抢占或真实GPU系统加速证明。
