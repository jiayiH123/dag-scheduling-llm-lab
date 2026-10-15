# 可复制给另一台电脑上agent的任务提示词

请先完整读取AGENTS.md、README.md、docs/HANDOFF.md、docs/LLM_INTERFACE.md和docs/problem_model.pdf。这是已有DAG调度研究的继续，不要从零重做宽泛选题。

目标：在固定通信延迟、硬预算、累积资源容量下，用一次完整神经网络forward和一次调度构造降低全局makespan。先复现GNN/BC/PPO及适配Greedy、Random、HEFT、PEFT、MSLBL、十规则和教师质量/时间对照，再探索LLM作为一次全图编码的结构化评分器，必要时离线生成更好的规则。不要假定LLM有效。教师是启发式搜索程序，GT为专家动作伪标签，非神经教师参数、非OPT。

先检查当前服务器GPU、显存、Python环境与我提供的本地model目录。跑check_bundle和knowncycle24评测，确认物理和基线一致。使用独立环境，不覆盖已有CUDA环境。若模型目录不能从当前上下文/本机找到，再向我询问路径，不要猜模型已存在。

第一阶段LLM只替换编码器并行输出[n*k,2]的a/w，保留相同teacher/mask/scorerange/固定HEFTdecoder，比较BC与BC+PPO至少3种子。普通generate/chatAPI输出JSON不是一次forward；只有聊天API时先说明接口限制，不能声称已完成一次forward模型。训练标签只用TRAIN，VAL选checkpoint，已提供TEST均为已看过诊断数据，最终确认要另锁新图。

若改任务优先级、预算机会值或解码器，需要同步给GNN对照。现有教师order是固定HEFT，需要新的TRAIN顺序示范或RL。优先查训练是否充分、固定order空间是否足够，别一开始只堆7B模型。记录RL相对BC增量、强baseline质量、完整推理耗时、参数/显存/训练工作，保留负例。teacher作为强搜索参考不能隐藏，也不是OPT；可研究接近teacher而更快的取舍，原用户期望超过teacher仍要追踪。

如果安装了autoresearch/ml-training-recipes/creative-thinking-for-research/brainstorming-research-ideas，按需要使用。用ara-research-manager只在任务末尾记录真实探索；没有skill时保留同等可审计日志。协议先提交再运行，更新当前假设和结果，禁止把BC收益说成RL收益或把已看过测试调参称为盲测。
