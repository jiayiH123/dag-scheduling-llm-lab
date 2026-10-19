# 2026-10-19模块实验增量入口

旧README、AGENTS和106个冻结文件保持原样；新增实现、证据和论文草稿从这里读。

- [结果与下一步](docs/MODULE_RESULTS_20261019.md)：73个执行训练臂，含六重复控制；97280训练候选；两轮小型公开开发实验。
- [公司电脑Agent接续提示](docs/MODULE_LAB_SERVER_PROMPT_20261019.md)：只pull、不push，优先扩独立公开DAG。
- [命令与接口](docs/MODULE_LAB_20261019.md)、[模块队列](docs/MODULE_QUEUE_20261019.md)。
- [英文论文工作稿](paper/iwqos_working_draft_20261019.md)：已填实际开发结果，尚未完成主实验。
- [完整增量证据](evidence/module_lab_20261019/aggregate.json)：原始summary、曲线、数据键及SHA在相邻目录。

公开数据只有8张TRAIN、4张VAL原始DAG，扩展为64/32配置；TEST未读取。校准GNN的0.75993C/HEFT仍落后MSLBL的0.65456，不是已获验证的最终方法。新模块CPU实测，CUDA路径待服务器确认。

运行`python scripts/verify_module_lab_addon.py`验证原冻结106文件和增量SHA。备份本地新增文件后`git pull --ff-only`，不覆盖已有服务器实验。
