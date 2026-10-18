# 直接复制给服务器 agent 的当前任务（2026-10-18）

你接手已有DAG调度研究，不要从零重做宽泛选题。先完整读取AGENTS.md、README.md、docs/SERVER_EXPERIMENT_PLAN_20261018.md、docs/HANDOFF.md和docs/problem_model.pdf。最新服务器计划优先于旧LLM接口；本轮无需下载LLM或寻找model目录。

目标：在固定可重叠通信时延、标量累计资源容量、硬执行费用预算下，用一次完整网络forward＋一次调度构造降低全局makespan，目标IWQoS。追踪超过教师、稳定BC后增量、强适配baseline。教师是启发式程序，不是OPT或神经网络；现有学生仍弱于MSLBL/十规则/教师，不能隐去。

方案：借鉴ReSched的任务/资源池交互，设计通信感知静态Attention；借鉴BOPO比较当前策略生成的完整可行方案。原论文没有完整覆盖我们的约束，需注明来源。ReSched四特征充分性、串行机器和每步forward不能直接继承。当前包没有新Attention/BOPO/CUDA trainer，请按计划实施，不能假报已存在。

先检查GPU型号/显存/卡数及CPU/RAM。“A00”只是我之前输入的简写，不要未经检查写成确定A100。建隔离环境，不覆盖系统驱动。运行真实入口server_preflight、check_bundle、8例smoke、known432复核。CPU参考在A100机器上仍是CPU；train_reference只是200BC/300PPO及180秒上限重放，不是主实验。

使用prepare_public_smoke按固定来源下载、校验并转换，先跑公开smoke基线。6条历史trace/24配置已被查看，不是新盲测，也没有TRAIN标签。主实验另选公开实例，先按底层trace划分再派生预算/链路，保留未见应用。统一平台并披露模拟假设；TRAIN生成教师，VAL选checkpoint，TEST冻结后确认，不能从TEST标签训练。

首轮固定HEFT构造顺序和LOCAL解码器，任务仍能并行。网络输出全部task–pool的a,w，硬预算reserve与资源日历由物理程序维护，动态mask/EFT不重复调用Attention。不同架构分别BC，同架构PPO/BOPO从相同BC起点出发；GNN/Attention×PPO/BOPO四组匹配实际候选数、训练工作、输入信息和anchor，附各自BC、自模仿与普通pairwise控制。旧GNN warmstart额外工作必须记录。

每条候选保留自己的可行集合、EFT、预算反馈及动作，用自己的真实前缀重算概率，不能借教师或赢家的前缀给另一条方案评分。区分实际采样B与筛选K。训练可多候选，部署只一次确定性构造。先查候选多样性和best-of-B空间，再看单次部署VAL曲线；pair loss下降不等于makespan改善。

docs/EXTERNAL_SOURCES_20261018.json有Git/论文/数据地址及固定版本；configs/attention_bopo_pilot.example.json仅是待实现配置契约，没有现成trainer消费。不要给旧脚本传不存在的--device/--candidates参数后声称跑通。单配置seed11排错后再11/22/33，依据VAL曲线和profile确定充分训练预算，不用几十秒试验判定RL无效。

协议先提交再运行。所有运行到新runs目录，不改vendor、冻结数据和检查点；保存代码/数据hash、B/K、真实解码与forward计数、优化步、显存、完整耗时、逐实例结果和负例。保留Greedy/Random/HEFT/PEFT/MSLBL/十规则/Teacher64。若装有autoresearch/ml-training-recipes则使用；ARA只在任务末尾记录，缺skill也保留同等日志。先自主检查服务器可读取的信息，再就真正缺少且影响运行的路径/资源提问。
