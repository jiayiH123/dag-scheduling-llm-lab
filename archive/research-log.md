# Research Log

- 2026-10-12：用户显式调用autoresearch继续项目，撤销上一轮仅实验方案的执行边界；最终目标仍为单次forward RL最小化全局makespan、面向IWQoS。
- 创建独立研究Git仓库 `experiments/autoresearch`，保留原始代码/先前实验；计划先运行H1-D1强基线诊断。
- 初始协议：120基础实例×3预算，共360配置；PEFT、汇合前瞻、单设备和局部搜索对照；预算CP-SAT与独立穷举核验。
- 当前没有可用Claude/OpenClaw定时会话触发工具；使用前台持续研究和文件检查点，不创建无效定时器。

- H1-D1协议提交7ef3898，随后8项验证通过；360配置9.55秒完成，2511启发式调度，720精确求解均OPTIMAL。
- 外循环1：主要残余受预算影响，不紧预算的中等通信头部空间不足。转向预算交换/单步搜索局部极值诊断，明确保留原makespan目标与预算约束。

- H1-D2协议提交ba11aee，9项检查通过。独立种子360配置27.66秒，3936启发式调度，720精确求解OPTIMAL；后续重验4656调度。
- 外循环2：预算交换使CCR1有效tight预算残余由15.837%降到4.853%，单任务搜索均已收敛。开始规划H1b小型CPU RL可行性pilot，尚无学习模型优越性结论。

- H1b-P1协议提交6b53f54；预算核心9项检查、actor/梯度3项检查通过。CPU完成11520训练轨迹和2160测试调度；360测试最优值全部证明，6459调度重验可行。每个策略12833参数，完整forward每图1次。
- 外循环3：冻结联合分数在短训练下明显输给order+EFT与强启发式；不支持H1b。先隔离训练时长，在全新测试种子上比较120/1200更新，再决定是否改动作解码。

- H1b-P2协议提交54d7ada，115200轨迹、4320测试调度，墙钟160.69秒（含强基线与精确求解）；360最优值全部证明，8622调度重验。前120步的6个checkpoint逐张量复现P1。
- 外循环4：joint长训练支持信号0/3；新测试集短/长均值OPT gap53.825%/54.276%，保留负结果。转向H2一次输出系数＋解析状态反馈，并加入同参数Frozen与无学习先验消融。H2草案未执行。
- 生成to_human/progress-001.html及4幅SVG/300dpi PNG科学图；逐图视觉检查通过。研究状态保存在文件；本轮完成后没有后台研究进程或自动触发器。

- 用户新指示：优先基于他人方法寻找真实研究问题，有思路阻滞时使用creative-thinking-for-research；autoresearch恢复，直接H2实现优先级降低。
- 重检索HBCS/MSLBL/BUDA/AILS/BCWS/critical-task/LPWS/AOM-SO/ERWS/HiGFRL与经典项目工期成本曲线；原论文／作者代码静态读，不运行外部源码。访问失败如实归档。
- creative-thinking F2/F3/F4生成并筛选11种角度；预算感知、学优先级、超图、普通分组均判为已有，保留协调映射与信用分配候选。
- C1-D1协议c71addf：邻域oracle增加mapping radius；初始测试命令缺PYTHONPATH导致导入失败，修正后原9＋新2项全部通过。360旧发现集14.08秒、新种子复验39.51秒；405＋460邻域均OPTIMAL，显著证书29／32，跨4非退化种子索引。
- C1-D2协议a781fa3：MSLBL与lex控制2项检查通过；360配置25.95秒；显著残余降至19，条件去通信／去预算／都去剩3／4／0，1977调度重验。
- C1-D3协议4840143：360配置18.08秒；DAG成组动作仅改善3／19显著参考，17个仍≥5%gap；全体gap0.969%，2520调度重验。结果保存f821e7f。
- HBCS下载35秒超时，按网络权限要求重试45秒仍超时；LPWS直接全文403。没有发消息联系作者或安装依赖。原始项目代码仍未修改。
- progress-002.html、3幅SVG/300dpi PNG、说明性四任务邻域证书与11张新文献卡已生成；链接验证与逐图检查完成。下一轮完整trace协议是草案，未执行；当前无后台研究进程。

- 继续执行C1-D4，协议9ab9058：完整58/52任务trace八配置，10秒参考搜索均未收敛；24邻域中4证明，全球0／8证明，322.12秒完成。实际相关性仍未确认，不将超时算作机制失败。
- C1-D4b协议7cf796b：2项松弛界核验通过，原资格／预算下穷举r0/1/2，八配置0.795秒；未给出高阶证书。112保存调度重验，6个参考可在radius2改善，3例为下界跨越证明。
- 后验分解说明案例：单独加速任务4/6/8超预算；单独放慢任务0恶化Cmax；联合四任务既可行又缩短Cmax。新机制候选需排除已有预算重分配／critical-task／种群搜索。
- 外循环8：继续问题细化，先关闭真实trace小邻域和时序基准，不新增协调RL训练。报告更新为外循环5–8，包含真实筛查未闭合结果。所有实验进程已结束，无后台任务。

- 2026-10-13用户恢复autoresearch，强调确切问题、baseline胜出和motivation。prepared kernel协议2a56b9e、3项检查通过；2400比较全相同，平均候选吞吐15.23x，所有新对照共用。
- C1-D5协议66e0733、2项预算补偿检查通过；8配置参考全部邻域收敛，97.67秒完成。预算补偿只1例改善0.0178%，无≥5%信号；关键任务donor主线在当前完整trace不成立，保留负结果。
- 外循环9：不追加该机制RL训练。下一实验分离全局顺序/映射/联合探索；先做相同候选模拟预算的5种子诊断，迭代搜索属于已有工作，不冒称作者算法完整复现。

- E1-D1协议20093fc：2项变异／限制臂／可复现／计数检查通过；120次搜索、720000候选，113.04秒。mapping在Genome有0.45%–1.37%median改善，order/joint无median改善，≥5%与联合优势门槛均0。
- 外循环10：不将搜索失败当作接近最优证据。构造全局assignment/依赖/通信/预算/负载必要条件下界；保留真实calendar可行上界。补检索2026 D-ERL、G-PPO和PPO-STGNN，按原文可读范围审计，不把多目标／动态场景直接当作一次forward预算基线。

- E1-D2协议1318c3f：2项检查通过；16个松弛解全OPTIMAL，6.01秒；相对已保存可行UB的最大可能改善0.132%–1.558%。这是原问题有效下界证据，非原问题全局最优证明。
- 外循环11：原Montage58/Genome52 fold剩余空间不足，转向新增完整应用而非增加同一RL训练。GitHub tree API沙箱和按权限重试均HTTP403；改读作者README，raw固定提交JSON可正常获取。Cycles67、Epigenomics41/73、SRA64已下载，尚未测调度质量；保留输入字节与哈希，平台参数保持不变。

- E2-D1协议b020252：输入转换1项检查（8旧实例）通过；16参考全收敛，643.44秒。10配置执行900000逃逸候选模拟；6配置被有效界排除5%空间。3例mapping/joint≥5%median改善（Cycles2参数变体、Epigenomics1），joint优势仍0。
- 外循环12：预算映射实质缺口在新增完整trace成立，但不把普通搜索作为创新。E2-D2协议f253092，AOM适配1项计数／重现／RNG隔离／物理检查通过；E2-D3条件机制协议22dbd66已经提交，等待作者对照后顺序执行，避免同时CPU负载影响限时收敛。

- E2-D2作者适配对照完成：3配置×5种子，90000TOTAL模拟；AOM更新也大幅修复Cycles与Epigenomics残余。普通anneal仅比其median低0.845%–2.917%，Epigenomics joint低4.323%；不以此冒称新RL/SOTA或同wall胜出。现在首次执行审计后的作者纯优化类，所有物理评价由共同内核完成，原代码快照未改。
- 外循环13：成熟高阶搜索已有能力，后续方向转向一次forward的质量/完整时延，而非首次发现预算交换。正在按已提交22dbd66执行全3例的条件机制和全部慢/快链路一致性审计。

- E2-D3 run001错误地assert固定映射/顺序SGS单调而失败，修正1d1f417并保留失败metadata。原执行时刻的可行性才保证参数单调；1项8对检查通过。run002全3例131.37秒完成；Cycles去通信仍有24.6%补偿收益、去预算参考60秒未收敛；radius2必要界均未认证显著3+障碍。
- 用户改变研究优先级：无需战胜全部baseline，先使一次forwardRL在所有约束下超过greedy/random/HEFT；借鉴开源和顶会代码，指标第一，继续autoresearch。此指示覆盖原novelty-first训练延后门槛。
- G1-P1协议7df2f67：2项模拟器/概率梯度/one-forward检查通过；正在准备800训练教师与新验证/测试分割，随后3种子BC300＋RL600更新。不复用旧测试调参，不新增依赖，原PPO-BiHyb未改。

- G1-P1完成：69.71秒，28800采样轨迹＋28800greedy baseline；2304学习输出均forward1且可行，6176保存调度重验。新test对HEFT整体4.302%/5.534%/5.316%，目标组8.615%/8.659%/10.750%，信号3/3；同场景Greedy/Random也平均胜出。BC已有效，RL只seed22整体增加、seed11下降、seed33回退BC，负例公开。
- 用户再指示持续训练→分析→调整→训练，用ara-research-manager记录，失败时用creative-thinking＋brainstorming。已读取研究记录skill（只在本轮完成后归档ara，工作中不读写ara），ml-training-recipes及experiment-loop用于下一控制循环；不执行破坏性reset，保存所有负结果。
- G1-P2预算状态PPO协议／实现已准备，2项checkpoint身份／预算特征／概率梯度／one-forward检查通过。两臂同14946参数、相同BC种子，新的验证50000/测试13000；运行前提交，不在旧test选超参数。

- G1-P2协议bc3722d完成：6run×400update，38400采样＋38400baseline，217.19秒。Static预算对HEFT整体5.173%/5.713%/5.844%，targetprimary3/3；Dynamic budget-state gate0/3且target输Static，保留负结果／不采用默认。4608评价全部forward1/可行，8480保存调度重验。
- 外循环16：StaticPPO提高整体均值但target仍弱于冻结BC，下一只改anchor.02→.2。协议1bdd490冻结，从同BC起点／同训练／同val条件出发，旧weak保存权重作对照，新的test15000；当前G1-P3正在训练。所有failures和checkpoints保留，ARA尚未读写，等本轮实验完成后epilogue。

- G1-P3完成：130.97秒，19200采样＋19200baseline，3456神经评价全部forward1/可行、7328保存调度再验。Stronganchor对target有小改善但额外2%门槛0/3，val/test整体弱于weak；保持weak.02静态预算默认。完整结果/负例/所有checkpoints保留。
- 外循环17：本轮3个学习实验完整结束；下一扩大DAG和No-extra-budget-ratio匹配对照的draft准备，不继续用小图旧test调参。默认seed11由最低validation选择，未挑test最好模型。报告003–006、数据CSV及11幅SVG/300dpiPNG完成，正在完成最终artifact核查，随后按用户请求执行ARA epilogue；所有实验进程已结束。

## 2026-10-13 resumed session: G1-P4 locked and launched

[ai-executed] Transport disconnection recovery verified all prior checkpoints/results and clean nested repo. Protocol commit2e514b9 precedes data generation. Critical fast-decoder equivalence passes64micro+24larger schedules; categorical logs/entropy/gradient vectorization agrees with old objective. Launch800train/216val/432test, disjoint latent DAGs, sharedBC plus matched INITIAL/LOCAL PPO3seeds. Only explicit score denominator differs; LOCAL retains input budget/mask. Source/code frozen; staged checkpoints and input journal enable recovery. ARA epilogue deferred until completed training/analysis.

- G1-P4完成329.23s：两臂baseline门槛3/3，PPO额外收益仍未达到2/3门槛。validation选择LOCALseed11/update250，结果未按test挑选。原globalphi随任务数缩小的train诊断记录为探索性解释。进入匹配采样的rewardbaseline控制，保持local评分和相同BC起点，fresh900000+test。

- G1-P5协议4e9b242＋分析预提交：借鉴POMO作者训练均值baseline及RLOO背景方程（保留primary快照，未执行外部代码），匹配K4采样／batch4／300update／180s两臂，从P4sharedBC开始，只改变奖励baseline。新test900000+，部署仍1forward。leave-one-out代数检查通过；训练已启动。

- G1-P5完成217.84s：RLOO相对BC门槛2/3，但相对匹配greedy0/3；保留旧validation候选。3seed原始advantage离散度下降但不是gradientvariance证明。再用creative-thinking F2/F4/F6＋brainstorming F1/F6/F7，记录16候选及3个最小可证伪方案；下一优先prefix条件baseline。两恢复训练循环结束，进入报告／ARA epilogue。

- [PM] 两个已完成会话的epilogue初始化ARA：31研究节点、4决策、16实验/审计节点、19可证伪claims、8heuristics；用户/AI provenance区分。25个SHA256证据绑定、2会话索引，YAML与本地链接核验通过。原问题PDF链接、负分支及尾部失败均记录；未声称ARA Seal。所有当前训练/分析进程结束，无后台触发器；下一3个方案未运行。

## 2026-10-13 session continuation: G1-P6

[ai-executed] Readstate/findings and autoresearch; originalsimulator/deploymentdecoderunchanged. TD-SCSTprimaryprior found andrecorded, no noveltyclaim. Protocol1dbbc01 precedesmeasurements: identicalphysical-prefixcompletionwork acrossGLOBAL/PREFIXarms, only per-decisionbaselinechanges;100updates/batch4/seeds11,22,33, fresh1000000+test. Independentcompletion/prefix/actionindependencechecks pass. No backgroundtriggeravailable;foregroundstagecheckpointscontinue. ARA notread/written during experiment.

- P6完成：Prefixbaseline额外门槛0/3，保持validation默认；两臂3272880总suffixplacements按seed一致，独立调度check全部通过。TD-SCST是已知设计，负结果保留。用户明确允许网络调整用ml-training-recipes，已读取该skill；下一针对通信表示，以零初始化、相同活跃参数edge对照隔离。

- P7协议9977540＋分析预提交：ml-training-recipes用于零初始化edge残差、同容量BASIS/DELAY活跃参数15202及BASE14946对照、单训练图学习检查、有限梯度、clip1、cosine及固定wallceilings。三种子×3网络，相同BC200/PPO300；fresh1100000+test。在36初始图forward比较中额外edgehead严格恢复旧权重输出，两扩展网络单图40步CE下降且edge权重非零；原deployment物理decoder未改。GiPH作者msg_func静态审读，仅借鉴，不跑作者DGL代码。训练已启动。

- P7完成：edge/encoding/tail/RL额外门槛均0/3，旧默认未被validation超越。新BASEseed11验证tie逐张量证实是旧policy复制，保留原checkpoint。网络调整按ml-training-recipes的零初始化、梯度/单batch/对照原则完整验证；负结果未删除。下一回到预算反馈评分，不增加网络容量。

- P8协议8d070d6：14946参数完全不变，LOCAL／STATIC平均allowance／DYNAMIC剩余平均allowance三评分模式；同P4BC零budgethead起点，复用原P4PPO300训练函数不改optimizer/anchor。fresh1200000+test，teacher分配不变只重记history特征。24随机20–80任务LOCAL完整调度/概率等价、三模式初始policy相同、B=Bmin边界完成检查通过；训练已启动。

### Cycle22 completed: G1-P8

AI executed9 controlled PPO runs, new fold1200000+, no hard-model changes. New normalization gates0/3; retained original validation-selected policy. Evidence:experiments/mean_allowance/analysis.md and results/run001/summary.json. Creative-thinking F2/F4/F5 now motivates TRAIN-only bound audit; no test-tail tuning, no model promotion.

### Cycle23 completed: G1-D5 representation audit

Creative-thinking F2/F4/F5 revisited the hidden bounded-score assumption. Training-only necessary reachability bound passes priority gate; no validation/test consumed. See experiments/score_capacity/analysis.md. AI proposes matched cap8/32 output-range continuation, not a user-authored novelty claim.

### Cycle24 completed: G1-P9

AI executed six matched BC+PPO runs after training-only D5 diagnosis. Tight25 gate2/3, all whole-gainCIs crosszero, overall/primary/tail0/3; retain original validation-selected checkpoint. Main objective still globalmakespan, not teacherCE or risk objective. Draft next direction: local EFT-gap scaling. No background scheduler/process remains after session completion.

### 2026-10-13 cycles25–27: user-inspired WeCAN-style problem discovery

User asks whether concrete baseline failure→mechanism→targetedRL can guide a paper. PrimaryWeCANv2 reread: immediate dispatch/skip reachability differs from our timelineinsertion. Executed I1-D1/I1-D2/I1-D3 with priorprotocolcommits; adaptive followup provenance explicit. Directupgrade-frequencygate fails, jointselectedutilitycontrol survives, cheapMSLBLportfolio narrows38 to1remainingcase. No new neuraltraining, no policypromotion, no scope/physicschange. Capture failedSingleDeviceassumption; baselineunavailable is not an automaticwin. Provisionalroute: reduce train510 and confirm independentprevalence before a futurevaluecurvehead. Allcurrent processes ended; ARA epilogue after result/archive/report verification.

## 2026-10-13 continuation: cycles28–30

[ai-executed] I1-D4 protocola5aca57:144unique induced subsets,19/29-task normalized-budget witnesses, no<=8case, CP-SATOPT458/521. I1-D5 protocol82a9d04:48newlatent/288configs,10178cheaperactionprobes;237>=5%HEFT/48portfolio/9joint/2threefactor signals, allnegativeconfigsretained. Selected-mask ledger exposes calendar→budget→tailcommunication cascade, not an immediate-dispatch skip requirement.

[ai-executed] P10 protocol1f01d79:128TRAIN/5600decisionlabels,7088offlinephysicaldecodes, supervised200+unchangedPPO300×twoarms×3seeds,14400sampled+14400greedy trajectories,289.95s. Auxmodule0/3 andnewRLextra0/3; defaultP4LOCAL11retained.6480testoutputsforward1/physicallyfeasible,12464savedchecks; exact-parent fallbackscountedwithoutnewimprovementclaims. Newfold1800000+HEFT16.518%/Greedy22.584%/Random47.179%;70HEFTlosses/worst+30.004%. Off-policy label mismatch is nextuntestedhypothesis, not diagnosedcause. Report012 andfigures35–37 completed; noactiveexperimentprocess. ARA epilogue follows now.

## 2026-10-13 cycle31: user asks constrained baseline gap and budget motivation

[ai-executed] Read actualHEFT/Greedy/cumulative/communication/reserve code; primaryHEFT/MSLBL/BUDA literature checked. I2-D1protocol6f28fea:12performance-independent reuseddiscovery instances×3budgets=36, CP-SAT10s/1worker/seed23;14OPT/22FEASIBLE,245.48s, allrecords/negativecasesretained. Historicaltwo360configsmallfolds allOPT;CCR1nondegenerate64pairedinstances perbudget: HEFToptgap9.986/26.935/31.912%, currenttimesaving7.549/18.756/21.029%. LargeHEFTtight25savinginterval38.284–52.881%;matchedincreaseversusnonbindingatleast5.149pp.7599savedchecks includeold/repeatedschedules. AuxiliarytenruleaggregateinitiallyincludedGreedy, correctedandbotholdsummary/CSVretained;mainbaseline/solvervaluesunchanged. No neuraltraining/defaultchanges. Report013/figure38 completed;ARAepilogue deferreduntilworkingdiagnosticcompleted.


## 2026-10-14 cycles32–33: user asks cycle24 provenance and teacher/baseline meaning

[ai-executed] Recovered progress010 and original generator/teacher code. Main800TRAIN and432TEST are self-generated synthetic graphs; public WfInstances24case transfer is separate. Reported average stability does not mean per-instance domination (seed11 HEFT76/432losses).

[ai-executed] I2-D2 protocol4a5dd7d: unchanged MSLBL and10rules on frozen432configs, reuse3seed neural outputs,5616checks/864exact schedule identities. Student mean relative C longer3.503% thanMSLBL and10.091% thanportfolio forhistoricalseed11.24.29s, no NN/solver/train. I2-D3 protocol390f2dd: unchangedteacher recipe on432knownconfigs,27648proposals/23464feasible;teacher7.263%betterthanMSLBL,student12.625%worsethanteacher.23.98s. All outputs retained; no test-label feedback or policy promotion.

[ai-suggested] Fix benchmark coverage and quantify teacher/student gap before another training recipe. Existing budget rules/teacher are stronger quality controls; one-forward quality/runtime value remains to be established. No background process or automatic scheduler.


## 2026-10-15 cycle34: strong teacher/OPT and data-dependent headroom

[user] Specifies final learning goals: surpass imitationteacher, obtain effectiveRL increment, outperform relevantadaptedcontrols; allowsGNN/LLM/training/component/data exploration. Asks aboutmodeledOPT headroombeforefurthertraining. [ai-suggested] Retainoneforwardandrigor: discloseeffectfulcomponents,freeze generators/testsplit, usefreshheldouttestforlatermethodclaims.

[ai-executed] I2-D4protocolc12dde6, reused720smallsourceOPT+36largerboundedconfigs,10.83s. TeacherRlower12.280%tight60/10.891%tight25;10/12and9/12>=5%certificates, bothgatespass.10548savedchecks/1512baselineidentities/252budgetphysics/240CCRphysics checks. No newsolver,NN,training ormodelexecution. MatchedsmallteacherR increaseswithbudgetpressure/CCRinselectedfixedcellsbutnotuniversallymonotone;allcellsandnegativecasesretained. Report015 includesknownfoldscope,variabledenominators andglobalOPT-versus-decoderreachabilityboundary. No backgroundprocess.

### 2026-10-15 — outer35, published gains and teacher semantics

User question (provenance:user): whether smallteacher-toOPTroom supports further work, how many percent relatedpapers improve, and whether teachermeans a neuralparametermentor or offline bestschedule labels. AI-executed: checked WeCANv2 primarytables, GiPH officialproceedings, AOM-SO authorpaper, originalteacher/BC/PPO code; wrote report016. No newtraining/NN/solver. Primarysource limitations retained inliterature note. AI-suggested: judge gains relative tostrongcomparators andcompute, not a universal threshold; quantify fixed-order mapping optimum andstudentcompressiongap before networkinflation. No user acceptance or newneuralresult inferred.

### 2026-10-15 — outer36, LLM roles and data contract

Provenance:user question: assess LLM benefit, representation changes/data conversion and decision role. AI-executed: read current12feature task-pool graph,2headactor andhardmaskSGSdecoder; read primaryReEvo/EoH/LLMsCanSchedule andofficialQwen/generation interfaces; appliedbrainstorming/creative frameworks; wrote13candidate ledger, report017 andunexecutedpilotdraft. AI-suggested: offline code-designfirst orparallelstructuredencoder foractualpolicy; preserve onefullforward/onedecode, graphsplit/commonphysics,matchedBC/RL/encoder/order controls. No experimental benefit/novelty or remoteaccess inferred. Earlier teacher-superiority re-ranking is AI-proposed, not userconfirmed.
