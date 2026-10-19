# 公司电脑Agent：本轮已经提供完整离线数据

先读根目录OFFLINE_DATA_README_20261019.md，再看本文件。用户最新要求是WfInstances和TPC-H两条数据线都具备离线实验条件。旧“需下载外部数据/仅12条公开pilot”的说明是历史状态。

1. 备份本地改动和实验目录，git pull --ff-only，运行python scripts/verify_offline_delivery.py。这里包含实际原始数据压缩分片、适配数据、TRAIN教师标签，不是链接清单。不要执行任何外部下载、git reset或clean。
2. WfInstances完整318记录和TPC-H作者154查询全部保留；TPC-H另外有1000个生成组合。已适配11776配置。主训练直接使用data/offline_ready_20261019下两个数据文件，分别为246/70和1600/200个TRAIN/VAL配置；详情见各自manifest。原800合成TRAIN仍保留为历史控制，不覆盖任何旧文件。
3. 先用TPC-H组合数据跑通较充分的BC及匹配PPO，同时以WfInstances独立开发集检查科学工作流表现。两者分别报告，不混算均值。开发阶段可以选择论文主线；设置确定后锁定TEST范围并报告该范围完整结果。当前没有新模型性能结论。
4. configs/offline_wfinstances_trainval.json和offline_tpch_composite_trainval.json为起始配方，含实际train_limit/val_limit和normalized_eft解码。先在公司本地提交训练协议及实际模型/资源记录，再训练，无须push。各架构从自己的BC检查点分叉；保留BC、Greedy、RandomMean5、HEFT、PEFT、MSLBL适配、portfolio和Teacher64。
5. 用户讨论的三个LLM对照是直接语言生成、文本SFT、结构化输入加并行a/w输出头。数据和GT已经可离线导出，但这些LLM模型尚未实现/训练。使用服务器已有本地权重；不从模型Hub下载。自然文本生成通常多forward，与严格单forward结构化策略分别记账。
6. scripts/export_llm_data.py支持--data指定新准备集，用相同TRAIN教师映射派生文本/结构化目标；VAL/TEST不得生成训练GT。完整物理数值可供各模型获得，不能把教师方案或baseline结果塞进评估输入。
7. 最大9805任务的完整图在稀疏库中，ready文件仅物化<=512任务子集以兼容当前稠密接口。不要把大图已下载解释为当前网络已通过9805任务显存/时延测试；拓展尺寸另开协议。需要更多pool/网络配置可完全离线运行prepare_offline_training.py。
8. 本机已验证全量源文件离线还原、哈希、约束字段、旧96配置等价、序列化TRAIN/VAL标签，以及两架构CPU单forward/概率重放；没有CUDA或训练后模型TEST验证。所有测试来源已按用户要求下载，但不自动用于模型选择。
9. 继续autoresearch记录，ARA仅任务结束后写。新结果保存在新runs目录，export_module_lab.py生成小JSON让用户复制回来；不需要从公司电脑push，不上传公司数据或权重。

TPC-H的通信量、平台算力/价格/容量等新增量必须在论文中声明为模拟。它是TPC-H-derived查询调度工作负载，不是官方TPC-H吞吐认证，也不是原作者完整问题的数值复现。

用户新增Halo论文（arXiv2509.02121v2）并质疑9805任务是否契合场景：先读docs/HALO_SCOPE_REVIEW_20261019.md。已附带data/halo_reference_20261019作者源码归档供离线阅读，但未生成Halo训练trace或部署其系统；不要把本包SQL stage当作LLM调用。当前WfInstances可用完整小中型图，<=200任务的真实执行记录有103条，9805只是全量最大值。主问题/规模选择应按应用及可验证假设决定，不按TEST胜负决定。
