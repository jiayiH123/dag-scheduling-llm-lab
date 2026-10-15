# Research Findings

## Research Question

严格一次完整策略网络forward，满足固定通信、累计容量和可加预算约束，降低全体任务全局makespan，并以IWQoS为投稿目标。

## Current Understanding

单次前向、学节点优先级、task-pool、skip、普通dense reward、预算分配、任务聚类与高阶超图均有先例；补检索的IWQoS2022 BUDA、LPWS2026、AOM-SO2026、HiGFRL2026是必须排除的直接／邻近冲突。

最新用户明确把优先级改为指标：一次forward RL在通信＋硬预算＋累计资源约束的同场景下稳定超过greedy/random/HEFT；无需击败所有检索到的baseline。允许借鉴开源与顶会思想，先试GNN，LLM也是可选而非必须。强搜索保留为教师和诊断，不再作为必须全部战胜才可训练的门槛。

已在完整Cycles/Epigenomics trace＋模拟平台发现可行预算映射改善，成熟AOM控制也处理大部分缺口。当前14946参数GNN＋解析EFT/硬预算解码平均超过核心baseline；仍保留P4LOCALseed11，最新P10新fold对HEFT16.518%、Greedy22.584%、Random47.179%，70/432输HEFT、最坏30.004%。P5–P10未产生更好的非重复validation默认模型，RL增量和tail稳健性仍弱；新fold不是模型进步。

WeCAN式问题发现已推进：train510压缩为19任务构造案例804→610（规则最好688、已证OPT458）；严格三消融负效用路径29任务943→749。新发现48latent/288配置上枚举每任务较便宜动作，237>=5%胜HEFT、48胜十规则、9预算/通信条件消失、2再容量消失。现有规则仍修复多数，离线枚举收益不等于神经可达收益。已有明确的资源空隙→预算mask→尾部通信位置连锁实例；不能将普通预算预留或rollout监督本身包装创新。

## Lessons and Constraints

- 原问题PDF逐时刻容量意图用Cumulative/事件扫描表达，不能照抄成对重叠求和。
- 主目标保持Cmax；预算为硬约束，不用加权能耗目标替换。
- 使用fixed通信主模型，shared_bus是另一个问题，分开研究。
- 同查询拓扑跨数据规模不构成独立结构泛化。
- 固定映射精确求解须使用该映射自身的可行horizon；自由映射最优makespan未必足够。
- 单次forward仅约束调度推理；训练过程的多次参数更新和额外模拟须记成本。
- 没有可用的会话定时触发器；状态文件用于连续工作与恢复，不声称已建立后台自动运行。

## Open Questions

当前重点：固定HEFT前缀反事实监督P10门槛0/3，下一草案比较冻结真实策略前缀/缓存logits后缀的动作成本标签，检验off-policy前缀错配解释（未证实）。P6前缀/P7边编码/P8平均预算反馈均模块失败；P9cap32仅tight25弱信号；未来价值曲线与EFT尺度草案尚未实现/训练。还需更稳健RL增量、tail修复、5训练种子、规模/设备泛化、完整新trace及校准平台。

## Outer loop 9: full-trace reference closure rejects the current donor-region story

2026-10-13。用户再次强调先找确切存在的问题、超过相关baseline、解释motivation；不把绝对“无人研究”当成必须证明的前置条件。

准备式SGS与原实现2400次完整trace候选比较，Cmax/cost全部相同；另200个随机映射与拓扑顺序逐任务/通信记录一致，3项内核检查通过。平均15.23倍是候选模拟吞吐的实现优化；旧路径每候选包含独立校验，新路径只对保存输出校验，不能作为神经推理或调度质量贡献。所有对照共用新内核。

C1-D5协议66e0733：8完整trace配置的单/双映射＋区域＋相邻无依赖顺序交换全部收敛。Genome宽预算参考从portfolio283867/283865降至242492；这比先前未收敛参考明显更强。区域预算补偿只在1/8改善80ticks（450448→450368，0.0178%），无≥5%信号，any-donor与positive-slack-donor结果完全相同。当前完整工作流不支持“必须放慢关键任务来支付通信组加速”的主线；合成证书仍有效，不能外推实际相关性。

下一轮按creative-thinking的表示重述：分开全局顺序探索、映射探索和联合探索，检查强局部参考的残余是否来自两者耦合。顺序扰动／预算感知迭代搜索已有AILS等先例，此阶段只作问题诊断与更强baseline，不宣称搜索本身新颖或一次forward RL已解决。

## Outer loop 10: controlled global escape has no substantial or joint-advantage signal

E1-D1协议20093fc：8配置×3臂×5种子×6000模拟，120次运行／720000候选模拟，113.04秒完成。3臂同一收敛参考、相同模拟数、硬预算和相同内核。逐配置按5种子median：mapping仅Genome改善0.453%、1.115%、0.461%、1.367%；order与joint全部median为0。所有预设≥5%门槛未通过，joint也未超过两限制臂；4个Genome配置仅各1/5种子joint同时胜过两者。

这拒绝当前联合探索叙述的支持信号，不证明顺序无价值或当前参考最优。相同模拟数不等于相同wall time或等价扰动分布；所有候选、预算repair和最终校验成本有记录。尚未声称超过AILS／AOM-SO作者方法。

下一步先给全局必要条件下界，以区分“小剩余空间”和“搜索还不够强”；同时补充不同应用的完整trace元数据。原两个trace只有2个独立结构，参数变体不是结构泛化。

## Outer loop 11: original trace fold has provably little remaining headroom

E1-D2协议1318c3f，2项下界/独立原问题穷举检查通过。全局assignment＋实际通信依赖＋硬预算＋每设备资源面积必要条件，去掉逐时刻calendar，16次5秒上限求解全部证明松弛最优，合计6.01秒。它不是原问题的最优解；松弛下界仍对原问题有效。

以E1-D1全部5种子所有臂中已保存最佳可行结果为UB（明确best-of-search，非单次部署）：Montage四配置最多再改善1.066%、1.558%、0.355%、1.190%；Genome四配置最多0.132%、0.236%、0.216%、0.204%。因此任意算法在这个fold相对这些UB都不可能再获得5%改善。这比“随机搜索没找到”更强，避免误把接近最优的对照当作需要复杂RL的新问题。

带/不带通信的必要条件最优值在8例相同；这只说明该松弛中通信未抬高下界，不证明原calendar问题通信完全无效。继续保持固定通信主模型，扩展应用结构而非修改约束来人为制造缺口。

## Outer loop 12: substantial complete-trace mapping headroom exists in new applications

E2-D1协议b020252，转换检查与8个旧实例逐字段完全一致。4完整trace／3新增应用族×4平台/预算配置，16个参考全部收敛；643.44秒完成。10个必要界仍允许≥5%空间的配置按预定门槛执行3臂×5种子×6000模拟；其余6配置按界跳过。

Cycles67/tight25慢/快链路：参考190193/190190，mapping5seedmedian139174/139414，降低26.825%/26.698%；joint145691/147978，降低23.398%/22.195%；单轮预算补偿降低24.567%/24.565%。Epigenomics41/tight25快链路参考120036，mapping108382（9.709%）、joint106812（11.017%）、单轮预算补偿107792（10.200%）。三个配置跨2应用族有实质信号，但两个Cycles带宽不是独立trace。order无≥5%信号，joint无相对两限制臂≥5%优势。Epigenomics73与SRA未提供该信号。

这是充分收敛既定局部邻域后、可行新解超过参考的真实机制证据；平台参数仍模拟，非测量硬件改进，也未证明这些参考在允许任意时序的半径2邻域全局最优。搜索／补偿不是RL新方法，不冒称击败所有相关作者方法。下一步对全3入选配置执行原作者AOM-SO更新的共同物理模型适配、然后条件机制分解。

## Outer loop 13: closest available author updates already handle much of the barrier

E2-D2协议f253092，三个按全筛查门槛入选配置×5种子×6000TOTAL模拟（含预算不可行），90000候选。原作者AOM-SO纯更新类不改，成本/SGS/累计容量/设备对链路/拓扑索引/共享参考种子/anytime上限为明确适配。median：Cycles慢/快140602/140602，较局部参考改善26.074%/26.073%；Epigenomics快111638，改善6.996%。所有run达到6000候选上限，候选可行率约80.4%/81.9%。阶段median时间2.828/2.721/.796秒，不含公共portfolio＋收敛参考。

普通mapping anneal median139174/139414/108382，比AOM适配median低1.016%/.845%/2.917%；Epigenomics joint106812比AOM适配低4.323%。这是同模拟数的探索性适配比较，不是同总wall time、原作者全模型性能或新神经方法胜出。AOM也大幅修复残余，故“成熟方法不会高阶预算交换”叙述不成立。

候选问题应收缩为如何在一次forward和可控完整解码成本下达到这些非局部分配的质量；高阶搜索本身已有。仍需跨新trace与端到端质量/时延前沿检验，再决定单次forward RL动作表示。

## Outer loop 14: conditional mechanism and user-directed metric-first pivot

E2-D3原协议22dbd66/run001先因错误单调性assertion中止：保持映射/优先级重新SGS并不必然因通信更快而更好。修正协议1d1f417/run002保留原执行时刻作可行证据；8对独立物理校验和已知greedy异常检查通过，原失败metadata保留。run002全3入选例131.37秒。

Cycles说明：B1509928、spare11172；未支付region C138888/cost1580372超预算；donors-only C206753比参考190193更差；联合5个改变C143469/cost1504832，改善24.567%。该region精确资金必要条件至少2个外部donor。去通信后仍有相同预算补偿收益；去预算60秒参考未收敛，因此不宣称预算局部障碍已消除。Epigenomics预算补偿去通信后仍存在，去预算参考收敛且无补偿收益。三个radius2必要界均未排除5%收益，不能声称完整trace3+全时序证书。

网络异常是邻近已有研究，不作首次发现：Epigenomics41宽预算慢参考66283，在快链路保持mapping/order重新SGS得到82509；原执行时刻在快链路仍可行C66283。Graham1966/1969、2025 anomaly分类、2026 DDE已有相关工作；本项目最优值的可行集单调性与具体greedy decoder单调性必须区分。

用户新指示优先超过Greedy/Random/HEFT，不以全检索SOTA为训练门槛；继续autoresearch、借鉴开放顶会代码。G1-P1协议7df2f67已提交，2项zero=HEFT/教师可行/概率与Torch/有限梯度/one-forward检查通过。800train、120val、360freshtest(offset11000)、24已知trace作为探索性跨规模transfer；300BC＋600全局makespanREINFORCE更新，3种子。模型选择仅val；BC-only与BC+RL配对，原图每次完整forward1，硬预算不变，正在执行。

## Outer loop 15: first positive single-forward GNN metric result, RL contribution mixed

G1-P1协议7df2f67完成，69.71秒；12833参数，3种子，14400BC训练样本次，28800采样RL＋28800greedy baseline轨迹。新test360配置：BC+RL选择模型对HEFT逐实例平均改善4.302%/5.534%/5.316%，Greedy7.564%/8.598%/8.324%，Random每实例5次均值40.078%/41.014%/40.662%。预定非退化CCR1紧预算64配置HEFT改善8.615%/8.659%/10.750%，信号3/3。平均Cmax之比的HEFT改善8.284%/9.704%/8.937%，不混淆两种聚合方式。

同freshtest比较BC-only，整体HEFT改善4.705%/5.157%/5.316%；RL仅seed22增益，seed11下降，seed33验证集选第0步（BC）。因此当前支持教师初始化、缓存GNN残差与解析EFT状态反馈的学习调度，不证明RL额外贡献稳定。40个潜在family/mode/随机实例簇bootstrap整体HEFT改善CI均大于0，条件于当前输入，非跨训练总体证明。

24已观察完整trace参数配置（6trace／5应用族）未参与训练/选模型，仅探索transfer：HEFT逐实例平均改善12.490%/12.339%/9.844%，但各有5/5/8例退步，最大退步约19.9%。平台参数模拟，不冒称独立确认／硬件部署。保存6176教师/基线/学习调度独立重验，2304神经输出全部物理可行且forward1；CPU组件p95微实例.71–.73ms、trace5.94–5.99ms，RPC另外报告。

用户要求autoresearch继续循环并用ara-research-manager做结束记录。下一G1-P2使用ml-training-recipes的梯度/学习率/有界循环与控制原则：配对初始预算比例vs剩余预算比例，one-forward输出额外系数，PPO＋教师锚定。仅动态vs静态预算隔离状态作用，未安排无预算PPO对照，不能单独归因PPO。新的验证50000/测试13000，旧模型同新测试重验，保持所有hard constraints与原makespan目标。

## Outer loop 16: static budget PPO kept, dynamic allowance not default

G1-P2协议bc3722d，14946参数，6run各400更新，38400采样＋38400greedybaseline，217.19秒。4608模型输出全部forward1/可行，8480保存调度重验。新test13000上initial-budget PPO对HEFT整体5.173%/5.713%/5.844%、目标组5.034%/5.360%/6.568%，primary3/3；remaining-budget PPO整体4.613%/5.445%/5.396%、目标5.029%/3.897%/5.398%，primary2/3。

动态vs静态同架构同训练控制：目标相对Cmax改善-.251%/-1.672%/-1.502%，预算状态门槛0/3。动态预算比例不作为默认；不把新模块概念当正证据。新test重验冻结BC对HEFT目标7.434%/8.110%/8.893%，说明PPO提高全体平均时牺牲紧预算组；与冻结REINFORCE的比较也分种子，不能混不同test数据。StaticPPO的val比dynamic全部好，但没有NoBudgetPPO匹配对照，训练与新头的单独作用未分离。

下一loop只改教师锚权重.02→.2（G1-P3协议1bdd490），保留14946参数／同BC起点／静态预算／训练数据／PPO/RNG/val条件。旧weak模型冻结对照，新test15000独立确认。用户要求指标优先，预算硬约束保留且baseline同约束；当前调整针对训练稳定性，不人为放宽策略预算。

## Outer loop 17: stronger anchoring trades average quality for small subgroup/transfer gains

G1-P3协议1bdd490完成：3run各400update，19200采样＋19200baseline，130.97秒；3456学习输出全部可行/forward1，7328保存调度重验。新test15000上stronganchor对HEFT整体5.064%/5.190%/4.676%、target9.609%/10.154%/10.530%；同testweak整体5.330%/5.507%/4.768%、target9.212%/9.584%/9.194%。

单因素strongvsweak的target相对Cmax改善.059%/.621%/.952%，额外≥2%门槛0/3；全部test反而相对退步约.47%–.52%，val整体三个种子也都弱于weak。因此保留weakanchor.02静态预算PPO默认，strong分支不推广。Knowntrace有1.180%–2.302%相对weak收益，但已观察trace不能推翻预定val选择规则。BC在此test目标11.273%/11.361%/11.757%，仍是重要对照，RL额外收益有改进余地。

目前完成用户要求的3次训练—分析—调整循环，有freshseed核心baseline平均优势；不是成熟论文结论。默认候选weights取budget_ppo/run001/initial_seed11.pt，依据3seed最低validationC/HEFT.930244选择，非test最好种子。下一draft扩展20–80任务和隔离预算评分/PPO作用，见experiments/large_dag_next/protocol.md；不继续在旧6–10任务test上调lambda。

## Optimization Trajectory

已完成原四轮诊断／RL实验，以及本轮发现集与新种子协调认证、强预算控制、区域／随机组控制；以下外循环区分机制、优化与学习证据。

## Outer loop 1: strong-baseline diagnosis

360个配置、2511个启发式调度；自由和固定映射的720次CP-SAT均证明最优。重验3231个调度（含精确解），预算/依赖/容量/通信均可行。

不紧预算时Portfolio-LS在CCR=1仅有0.451%平均gap；tight25为13.201%。无通信tight25仍有18.049%平均gap。预注册混合预算门槛通过，但不能据此支持通信单独造成神经方法机会。

主要残余来自映射，而非时序。下一步检验充分收敛的单任务迁移、预算成本价格扫描和双任务预算交换；尚不开始复杂神经训练。原H1在当前不紧预算微实例上证据较弱，预算耦合H1b作为新机制候选。

## Outer loop 2: budget exchange

独立种子360配置、3936启发式调度；720精确求解均OPTIMAL，重验4656调度。中等通信、非退化tight预算64配置：Portfolio gap15.837%，单任务充分收敛仍15.837%，Exchange-LS降为4.853%（残余降低69.36%）。全部单任务搜索达到无改进，避免把轮数不足误认为机制。

双任务预算交换是可解释的简单修复，但仍存在少数高阶映射残余。两任务无依赖例子[12,8]、速度[1,2]、费率[1,4]、Budget32说明：先把短任务放快节点使长期单迁移不能改善，联合交换可使makespan12降至8。此例为实现验证，不能作创新或总体效应证据。

当前决定：H1原通信独立叙述证据不足；将预算-通信耦合的一次前向联合策略记为H1b，先做小型RL可行性pilot。必须与成本价格、双任务交换等强对照比较，报告速度与质量。

## Outer loop 3: frozen single-forward RL is not sufficient in the short pilot

H1b-P1：2000训练配置、120验证配置、独立种子360测试配置；两种策略×3训练种子×120更新×batch16，共11520采样轨迹。测试2160个学习调度全部可行且每图恰一次完整forward；360测试最优值均由CP-SAT证明。重验6459个调度（含基线与精确解）。

order三种子平均OPT gap为34.392%、31.476%、31.648%；joint为46.980%、53.652%、48.978%。joint虽比随机初始化改善，仍比order更差，不能支持H1b。order/seed22的验证最优checkpoint为初始化，负结果完整保留。原型p95组件时间约0.82-0.85ms，仅为6-10任务CPU微实例，不是部署延迟。

当前解释候选：120次更新不足；静态联合logits只能通过预算掩码适应状态，可能难以分配预算。二者尚未隔离，不能将pilot失败归因于一次forward本身。下一步只改变训练时长，在新的测试种子上对比120与1200更新，再决定是否进入H2状态反馈。

## Outer loop 4: longer frozen-score training does not resolve the gap

H1b-P2：另一个独立测试seed_offset5000，360配置；同一模型继续训练至1200更新，记录第121步学习率重启，参数、动作空间与解码不变。前120步的6个最佳checkpoint与P1逐张量完全相同，确认对照复现。共115200训练轨迹、4320测试输出；8622调度重验可行，360最优值均证明；每测试图完整forward为1。

三个训练种子均值：order短/长OPT gap32.767%/33.715%；joint短/长53.825%/54.276%；新测试集Exchange-LS为1.973%。CCR1非退化紧预算组Exchange为5.117%，joint长为51.678%–61.681%。三个joint长模型均输给同种子order，预设支持信号0/3，且均未优于HEFT。

验证改善不保证新测试改善：joint长只在seed22明显改善测试，seed11/33变差。短/长测试差是同一新测试集上的配对结果；不能用P1与P2的不同测试数据直接比较。结果不证明单次forward不可能，只拒绝当前冻结评分实现及训练设置的优越性。

当前决定：停止给同一冻结架构增加训练预算，转向H2一次输出系数＋解析状态反馈；须与参数匹配Frozen和相同先验的无学习规则比较。下一轮草案在experiments/h2_state_feedback/protocol.md，尚未运行。H3当前使用HEFT动作无关baseline但没有其因果消融，仍待检验。

## Outer loop 5: user-directed problem discovery and coordination certificates

用户明确要求基于基础场景，从别人方法寻找真正存在的未解决问题；此优先级高于直接实现H2。重新检索预算文献，发现预算／关键任务不足／学优先级／分组并非新问题，延后泛型状态反馈训练。

Creative-thinking F2/F3/F4将问题重述为“改善Cmax所需协调改变的最小映射数量”，同时用经典项目工期—成本曲线排除表面类比。必须保留makespan、硬预算、容量和一次完整forward，不借增加DVFS、动态到达、链路竞争制造新问题。

C1-D1发现集360配置：充分收敛pair后57个3+证书，其中29个gap≥5%；新种子360配置：63个证书，其中32个gap≥5%；各覆盖4非退化种子索引。405＋460邻域CP均OPTIMAL，允许任意时序，排除一／双映射＋时序能改善的解释。原pair轮数限制只解释5／2个残余；固定映射精确时序解释5／9个剩余配置。

构造的五任务无通信例子[8,8,8,4,4]预算56，参考映射C8/cost40，0..4映射不能改善，5个改变达到C4/cost56；全部32映射穷举核验。该例仅验证机制／oracle，不能作网络研究创新依据。

## Outer loop 6: strong simple controls narrow the candidate

C1-D2在同一新种子360配置上，加入MSLBL原Eqs8/9/22预算预留适配版，及充分收敛的lex(Cmax,cost)一／双迁移。MSLBL+pair在32配置胜过原pair；lex再改善5配置，另有39个成本中性时间迁移。全体gap1.939%降至1.082%；显著3+残余32降为19，仍跨4种子、4族，其中CCR0/1/5为2/13/4。

在这19个固定参考映射上做条件干预并精确优化时序：去通信剩3个局部障碍，去预算剩4，两者去掉为0。12个案例中单独去掉任一因素都消除障碍。此结论是选中映射的局部障碍分析，不能冒充所有DAG的总体因果作用；干预也改变了全局目标值与难度。

全部诊断证明，1977个保存调度独立重验。预算公式有实际链路／eligibility／累计日历适配，未声称作者实现原样复现。尚无RL优越性结果。

## Outer loop 7: structural group control does not eliminate every tail

C1-D3依据DAG路径区间和前驱／后继星生成3..n成组迁移，固定候选每项只模拟一次，不执行神经模型。对照5个种子的同数量同规模随机组；可行模拟数不同，已记录。全体gap0.969%，19个显著参考中仅3个改善、2个改善Cmax≥5%；17个仍有≥5%OPT gap，原19个选中案例平均gap11.08%降至9.41%。17个不等于围绕新映射重新认证的17个高阶障碍。

平均65.625候选、32.981可行模拟、区域搜索7.779ms（不含之前强种子），随机组全体gap1.072%–1.082%。2520保存调度重验。一个按最大CCR1 gap选择的9任务说明例，邻域r0/1/2/3均555，r4达到457；预算1446，成本1264→1428。该案例单独精确补证，不作为总体代表或论文主结果。

下一方向：完整真实trace＋模拟平台的实际相关性门槛，及反复区域／中性迁移与同时间搜索控制。只有通过，才训练一次forward协调bundle策略，与同参数singleton、同候选非学习规则比较。候选问题成立于合成基础模型；“无人解决”、实际价值与IWQoS创新均未确认。H2、H3暂不抢跑。

## Outer loop 8: full-trace screening is inconclusive; simple-neighborhood closure comes first

C1-D4协议9ab9058：完整Montage58和Genome52×2带宽×2预算，共8配置。原trace依赖、时长、文件字节保留；设备相对速度、单位slot需求、容量、成本与带宽明确为模拟假设。10秒预算的lex/region参考均未收敛；CP邻域0/1/2各10秒、全局5秒，322.12秒完成。24邻域仅4个OPTIMAL（两Montage宽预算的r0/r1），全部8个全局仅FEASIBLE；没有高阶协调证书。

C1-D4b协议7cf796b：穷举全部可行半径0/1/2映射，以固定映射最长依赖路径＋设备累计duration×demand/capacity负载作必要下界；2项独立例子／CP核验通过。8配置全部穷举仅0.795秒，但松弛未闭合。必要界差不能等同最优值差。

112保存调度独立重验。radius2以内可行结果改善6／8保存参考；3例的radius2上界低于固定映射下界，证明存在小邻域映射改善。当前实际相关性门槛未通过且不作失败证据：参考没收敛、证明没闭合。不能把solver超时或改进候选当作高阶机制证据，下一轮优先小邻域／时序收敛、改进必要界、同时间适配搜索，而非扩大RL。

选中合成例进一步后验分解：任务4/6/8加速额外成本300，spare182，不协调则1564>B1446；任务0放慢释放136但单独精确Cmax恶化为682，联合4映射使cost1428/C457。这支持“局部变慢以支付整体通信组加速”的候选机制，不支持首次发明预算重分配，也不证明当前RL训练失败的因果解释。

## Outer loop18: larger-DAG baseline gains, but added RL signal still weak

G1-P4协议2e514b9，20/40/80任务800训练图、24验证图/216配置、48新测试图/432配置，14400采样＋同数greedy反事实，329.23s。两评分臂3/3通过核心规模门槛：HEFT整体13.764%–14.891%，CCR1紧预算20.503%–22.128%；Greedy20.11%–21.20%，Random45.05%–45.94%。5184神经输出均一次forward，11168保存调度重验；原模型与所有约束保持。

BC已约13.5%–14.1%整体HEFT收益；PPO额外>=1%门槛INITIAL1/3、LOCAL0/3，GLOBAL预算分母优于LOCAL额外>=2%门槛0/3。validation-selected候选LOCALseed11，testHEFT14.891%、Greedy21.199%、Random45.940%，tight25 29.409%、nonbinding.415%。负例73/432、最坏HEFT退步30.99%，不宣称逐例支配。下一针对信用分配，用匹配4采样的greedy/RLOO baseline控制继续训练，不增加推理forward或bestK。

## Outer loop19: matching samples limits RLOO attribution

G1-P5协议4e9b242/7ee8e6d，新48图432配置seed900000+；两臂匹配K4/batch4/300update，28800采样＋7200greedy，217.84s。RLOO相对BC1.015%/1.770%/.398%，>=1%门槛2/3，但seed11的图级CI跨0；相对匹配GREEDY仅.335%/.346%/.176%，全部CI跨0，模块门槛0/3。两臂核心baseline门槛均3/3，保存5184神经输出forward1、11168调度重验。

所有新validation差于旧LOCALseed11，保留原候选；它在另一个freshfold对HEFT15.010%、primary22.593%。不因测试略好而更换模型。Creative-thinking＋brainstorming将下一问题收敛为状态条件credit和随n稀释的预算评分；三个可证伪方案见新credit笔记。此阶段不增加盲目训练轮数，也未宣称RLOO首次创新。

## Outer loop20: prefix-conditioned reward baseline fails the module gate

P6协议1dbbc01：6run×100updates，2400采样、3272880后缀taskplacements，174.18s。两臂同seed放置次数完全匹配；560completion保存调度独立验证。PREFIX相对GLOBAL-.747%/-.073%/-.979%，模块>=1%门槛0/3；对BC仅1/3过门槛。adv std下降但zero比例增至32%–35%，不能把原始adv离散度当gradientvariance/指标改善。5184神经输出均完整forward1且无训练补全泄入推理，11168保存调度重验。保留旧P4LOCALseed11（freshfoldHEFT14.881%）；接下来有对照地改通信edge表示。

## Outer loop21: explicit edge features fail matched controls

P7协议9977540/731e4cd，9BC200＋PPO300运行，377.90s，21600采样＋21600greedy。BASE14946／BASIS-DELAY15202活跃参数。DELAY对BASE≥1%、对BASIS≥1%、高CCR紧预算q90tail≥25%三门槛全部0/3；三个臂RL额外≥1%对BC也0/3。更丰富edge编码未带来稳定指标收益，不扩大网络。9072神经输出均forward1，15056保存调度重验。P7 BASEseed11与旧P4LOCALseed11权重逐张量完全相同，验证tie不是新改善；保留原文件。它在第四个freshfold对HEFT15.222%，82/432退步、最坏41.36%。下一只改每任务平均预算评分，先静态、再剩余动态，对照LOCAL。

## Outer loop22: per-task allowance scaling fails despite exact LOCAL reproduction

P8协议8d070d6，9run×300update，21600采样＋21600greedy，335.26s。STATIC/DYNAMIC对LOCAL、DYNAMIC对STATIC三个指标门槛均0/3；两个tail联合门槛0/3。STATIC/DYNAMIC的RL额外>=1%均0/3，LOCAL1/3。seed11动态tail下降29.04%但整体退步1.961%，不算成功；其他种子tail没有复现。6480神经输出均forward1、12464保存调度重验。保留旧P4LOCALseed11，本freshfoldHEFT16.131%、primary24.227%；76/432退步、最坏27.923%。LOCAL严格复现旧结果，不能把新fold数值较高当成改进。下一先在800训练prefix上做评分范围可达性审计，避免无依据加参数。

## Outer loop23: bounded scores exclude some feasible teacher actions

TRAIN-only G1-D5协议b3f0858，800图36540teacher-prefix决策。即使a,w独立取[-8,8]最有利值，1878决策(5.140%)／117图(14.625%)仍不能选择教师动作。1858集中tight25，跨CCR0/.1/1/5；不是仅通信的问题。cap32解析放宽只余12，但尚非训练或makespan收益。此发现针对当前评分的给定prefix；不证明所有更好解不可达、不证明RL失败的唯一原因。下一同参数cap8/32匹配BC+PPO，保留一次forward／一次解码。

## Outer loop24: score expansion helps tight25 narrowly, not overall or tail

P9协议abca367，CAP8/CAP32同14946参数，6BC200＋PPO300全完成；14400采样＋14400greedy、297.63s。CAP32对CAP8整体+.093%/-.048%/+.617%，CI全部跨0；tight25+2.474%/1.321%/3.014%，联合门槛2/3，whole/primary/tail0/3。RL额外两臂均1/3。神经输出虽可达32，但选定PPO最大仅9.40/7.14/8.22，饱和比例0，解析能力改善未自动转为策略收益。6480神经输出forward1，12464保存调度重验。旧P4LOCALseed11仍最佳validation，本foldHEFT15.583%、Greedy20.966%、Random46.360%、primary22.633%；76/432输HEFT，最坏27.648%。后续优先局部EFT差尺度与残差平衡，配对保留zero-scoreHEFT和一次解码；不加盲目参数／训练轮数。

## Outer loops25–27: WeCAN-style problem-first investigation narrows the candidate

用户建议按WeCAN的反例→机制→学习路径找论文idea，主目标/默认模型不变。I1-D1协议a41b4aa：3任务预算反例精确OPT38，Greedy56/HEFT53；800TRAIN首分歧反事实中187例局部更慢更便宜且整体改善>=5%，但严格直接unlock后继升级机制0/800，未通过。I1-D2协议2501173/22f0be3：固定187正例控制；51例在去通信或放开预算后动作效用分别<=0，17例去容量也<=0（后验）。SingleDevice可行692/800，38例联合条件且对可行Single>=5%。I1-D3协议7d05fbd：这些38例中，反事实对MSLBL>=5%仅1，对十规则portfolio>=5%也仅1。多数已有廉价预算规则可修复，不能宣称普通预算预留首次创新。

剩余train510是40任务randompacking/CCR1/tight60，局部EFT57→106但全局1054→774，MSLBL/portfolio903；同动作去通信/预算/容量的效用-6.686%/-16.963%/-17.339%。这是选择的可行反例，不是总体频率、最优性或新网络结果。下一先最小化/解释该样本，在新发现集和完整工作流检验代表性，再定预算条件下游价值一次forward策略；EFT尺度工程草案仍未运行。4955保存调度校验通过（含重复控制）；800原解码等价与800教师完整重放通过。SingleDevice恒可行assert失败已修正并保存失败attempt。

## Outer loops28–30: independent action gaps survive cheap rules, fixed-prefix supervision fails

I1-D4a5aca57 reducedselectedtrain510to19normalized-budgettasks: HEFT804/portfolio688/CF610/OPT458; strictnegativefactorpath29tasks943/871/749/OPT521. Both single-deletionminimal undertheirlockedpredicates, not globalminimum; target<=8failed. Ledger shows earlyresourcecalendar changes alterlaterbudgetmasks andtailco-location.

I1-D582a9d04 fresh48latent/288configs:237slower-cheaper>=5%HEFTgains,76MSLBL,48portfolio,9budget+commconditionaldisappearance,2alsocapacity. Offlinefullactionprobes, not neuralpolicygains; allnegativecases/counts archived. CCR0/1/5portfolioresidual4/14/30 motivates communication-sensitive futureallocation without universalcausalityclaim.

P10counterfactualtargetpilot1f01d79, unchanged14946actor, sixSUP200/sixPPO300runs. FixedHEFTprefix/globalreturnauxgate0/3; RLextra0/3. SupCFall3selectstep0; exactclones11/33arefallbacks. Seed22PPO-CF/matchedteacher−.441% CIcross0. Originaldefaultunchanged; currentnewtestHEFT16.518% with70losses andworst+30.004%. Learned-policyroll-in/suffix labels are an unexecuteddraft, genericrolloutlearninghasLOLS/ExItprior. No newinferencearchitecture orpostprocessing search.

## Outer loop31: budget constraints increase avoidable loss in matched diagnostics

Userasks currentbaselineconstraints/OPTgap andwhetherbudgethelps identify a meaningfulproblem. HistoricalsmallCCR1/nondegenerate64pairedbases across2folds: HEFTgap9.986→26.935→31.912%(eta1/.6/.25), reduciblecurrentCmax7.549→18.756→21.029%;Greedy9.570→24.628→25.026%reducible. Do not invertmeanratios. I2-D1new36budgetconfigs/12reuseddiscoverybases (all4families×20/40/80,pool/CCR1/rep0) CP10s14OPT:HEFTtight25trueheadroommeaninterval38.284–52.881%, matchedincreasefreebudgetatleast5.149pp. MSLBLtight25headroom15.632–35.583%indicatesknownrulesrepairpart. Simulatedprice/speed [1,4,16]/[1,2,4] creates costtradeoff; notuniversalprovider/hardwareevidence. BudgetalreadyinPDF, nofirst-budgetproblemclaim (MSLBL2017/BUDAIWQoS2022prior). Current default unchanged; learned-prefix label draft remains nextunexecutedmethodcontrol.


## Outer loops32–33: known budget methods beat the current student on the same fold

I2-D2 protocol4a5dd7d closes a full-fold evaluation gap: cycle24 frozen policy versus MSLBL-adapted and existing ten-rule portfolio on all432/48latent known test configs. Seed11 neural gains -3.503% and -10.091%; all3seeds negative points, seed22/MSLBL CI crosses0.5616savedchecks/864exactHEFT-PEFTidentities. No new NN calls or model tuning. Basic HEFT/Greedy/Random wins remain valid but do not establish superiority to budget-aware controls. Default remains a neural reference, not the best known overall method.

I2-D3 protocol390f2dd calls unchanged training-teacher recipe on all432 known configs, diagnostics only: teacher overMSLBL7.263%, overnativeportfolio2.204%; student seed11 is12.625%longer relative toteacher (87/25/320).27648remap proposals,23464feasible,432validatedteachers. No CP-SAT or NN call; no TEST-label training. Original800train teachers/36540labels are heuristic-plus-local-search, not solver-optimal demonstrations. Quality compression loss is observed; causal representation/learning explanations remain untested. Before further training, budget-aware comparison and BC/RL attribution must be maintained. Existing C40 policy-prefix recipe remains a draft.


## Outer loop34: strong methods and the teacher retain modeled OPT headroom

I2-D4protocolc12dde6, all720smallsourceOPT and36largerknownboundedconfigs. Unchangedteacher64/newrules update onlyfeasibleU, no newCP/NN/training. Largertight60/tight25teacherR=(C-C*)/C meaninterval12.280–27.048%/10.891–31.969%;10/12and9/12casescertified>=5%,bothlockeddiagnosticgatespass. MSLBL19.568–33.644%/15.710–35.583%;10rules14.097–28.341%/11.916–32.755%. More room exists than justweakbaseline gaps, butglobalOPTmayneed schedulingorders/timings outside currentfixedHEFTdecoder. No promise RL/LLM can capture it.

Dataeffect matched80physicalsmallbases acrossCCR/budget,64nondegenerate withprovenOPT: Teacher tight25R atCCR0/1/5=2.637/6.342/9.593%;CCR1budgetsnonbinding/.6/.25=.552/4.845/6.342%. Notuniversalmonotonicity:nonbindingCCR0→1teacherRdecreases;MSLBLCCR1tight60→25decreases. Changegeneratorbottlenecks before freezingtest, notfilterbytestwins. Currentevidence supports mechanism-driven stress distributions, not learned-method-specific superiority. Userreaffirmsteacher/RL/adapted-controlsuccessgoals andflexiblearchitectures;result-affectingcomponents mustremain disclosed. Currentactorunchanged;C40notexecuted.

## Outer35 — gains in primary papers and teacher terminology (2026-10-15)

AI-executed literature/code audit, no new experiments: WeCAN greedy TPC-H gains8.47/10.05/8.48% versus IPPTS; RI-W sampled256 gains6.22/6.15%, and raw table means imply1.16/3.42% versus PPO-BiHyb. GiPH headline up-to30.5% is not a HEFT gain (HEFT comparable). AOM-SO reports overall43.03%,but41-task MD vs originalSO7.07%; these are different comparators/models/budgets. No universal publication threshold or OPT-room estimate follows.

Our original teacher is a heuristic/search program supplying36540hardactionpseudo-labels over800trainconfigs, not a separatelytrained neural teacher or source of transferred parameters. BC warms up student parameters by negative actionloglikelihood; retained PPO also has.02teacheranchor. Causal effect of thatanchor on surpassingteacher remains untested. Small exact OPTroom cannot be generalized to larger main distribution, and globalOPT room differs from fixed-order mapping room. [Audit](literature/improvement_scale_teacher_20261015.md), [report016](to_human/progress-016.md).

## Outer36 — LLM role and input/output design, untested (2026-10-15)

User asks whether an LLM helps, what data conversion is needed and what decisions it should make. Brainstorm/problem-first/simplicity/decomposition and creative representation/constraint frameworks yield13retainedcandidates. Current syntheticnumericalDAG has no natural-language tasksemantics; replacingthebackbone neither changes fixedHEFTreachability nor fixes creditassignment. ReEvo/EoH support offline heuristicprogramdesign; authorLLMsCanSchedule paper supports finetuned textJSSP as a different setting, not exactmodel superiority.

Proposeddirectpolicy: fullinstanceprefix then known task/pairqueries,onecausalbackboneforward andparallel a/w(+priority)heads, exactphysicalmask/SGSdecode. Ordinarygenerate is multi-forward even withKVcache. Original36540teacheractions can superviseassignmentheads; fixedHEFTlabels cannot teach betterorders. Firstisolate encoder under identicaldecoder, then GNN/LLM x fixed/learnedorder if justified. LLMselectingonefixedrule cannotbeat perinstancebest ofsamecandidate/decoder set. No newLLM/modeldownload/server/training/NN/solver;defaultunchanged. [Roleaudit](literature/llm_roles_20261015.md), [draft](experiments/llm_role_pilot/protocol.md), [report017](to_human/progress-017.md).
