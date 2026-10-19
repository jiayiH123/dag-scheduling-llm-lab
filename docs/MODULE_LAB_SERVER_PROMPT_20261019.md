# 给公司电脑Agent的继续任务

目标：通信/累计容量/硬预算下，一次forward调度DAG，最小化全局makespan。优先搭可用架构和论文原型。公司电脑只能pull，不能push。

1. 备份本机新增代码、runs、exports和检查点，记录git status，再git pull --ff-only。冲突先保留本机副本，不reset、不clean、不覆盖。AGENTS.md是冻结说明；本增量状态见MODULE_LAB_20261019.md、MODULE_RESULTS_20261019.md和MODULE_QUEUE_20261019.md，沿用其研究纪律。
2. 先读结果。本机完成73臂，含六重复控制；公开只有8TRAIN/4VAL底图。修复EFT先验尺度后GNN BC为0.75993C/HEFT，仍落后MSLBL0.65456；RL尚未稳定超过较强BC。冻结参考P4 LOCAL11没有被替换。
3. 先文件校验、原冻结审计及GPU preflight，再checks-only。新增实现隔离在daglab/module_lab及scripts，不覆盖你已有Attention/BOPO代码。CUDA尚未本机实测，先查一次forward、物理可行性和每候选自己的prefix概率。
4. 复现公开pilot的命令见MODULE_LAB_20261019.md第4节。原评分与normalized_EFT分别配置，不混用检查点。只下载TRAIN/VAL；source Git blob/SHA必须匹配。gzip头和计时允许跨机器不同。
5. 下一步先扩独立公开TRAIN/VAL执行trace，不只是为这8图扩pool/预算。先固定source commit、路径和拓扑组，核查旧开发数据；预算/网络/需求假设公开。新TEST冻结后保持未读取，设置确定再用独立评估器。不要把旧known-public-smoke当盲测。
6. 最小模型保留校准GNN+LayerNorm，补参数量约3万的GNN与Attention对照。充分BC后从各自同一检查点分叉PPO和一个温和正则控制。先不扩十几种loss。保留BC、Greedy、RandomMean5、HEFT、PEFT、MSLBL适配、portfolio、Teacher64；未超过强对照就诊断差距。
7. paper/iwqos_working_draft_20261019.md已有模型、方法和开发表格，可以边跑边写。最好运行明确标为最佳开发运行，已有均值和负例保留；step0不是RL增益，借鉴模块注明来源。
8. 每次先独立commit协议再运行，只本地commit，无须push。新runs目录不覆盖；scripts/export_module_lab.py导出小JSON供用户复制回来。原summary、曲线、数据清单、检查点留本机。不要上传公司资料、凭据或权重。

本轮不要求重跑全部旧合成筛选。增加独立工作流优先于堆大模型。设置锁定后，仍需最少几个训练种子和一次未见数据确认；先完成上述最小闭环，不做无目的扫参。
