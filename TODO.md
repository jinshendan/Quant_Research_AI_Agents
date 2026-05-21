# TODO --- 个人版 Hedge Fund Quant Research Lab 路线图

本 TODO 面向个人自用场景：把当前 Agent 系统从“模板因子日报工具”升级为
“个人版 Quant Research Lab”。目标是尽量接近量化公司 / Hedge Fund 里
Quant Researcher 的研究流程：系统地产生候选因子，严格验证，淘汰无效信号，
组合少数有证据的 alpha，再服务于股票观察、交易前判断和复盘。

重要边界：

- 本项目不是自动交易系统。
- 本项目不能保证赚钱。
- 本项目的目标不是让 Agent 直接喊买卖，而是帮助我们用更专业的研究流程减少主观拍脑袋。
- 任何进入实盘观察的信号，必须经过样本外、稳定性、成本、风险和人工复核。

## 当前定位

当前项目已经完成基础研究管线：

- 真实 A 股日线 OHLCV 数据接入、清洗、交易日历对齐、DuckDB 存储和缓存。
- Daily research pipeline：
  `DataAgent -> FeatureAgent -> BacktestAgent -> CriticAgent -> DailyRankingAgent -> MemoryAgent -> ReportAgent`。
- 基础因子模板、因子矩阵生成、ranking transform、rolling feature。
- 单因子回测：IC、RankIC、多空收益、Sharpe、Drawdown、交易成本后 net metrics。
- Benchmark quality gates 和 CriticAgent，可以拒绝低质量因子。
- 每日候选股排名、研究报告、记忆存储、dashboard 和中英双语输出。

当前项目还没有达到 Hedge Fund Quant Research 标准，核心缺口是：

- 因子主要来自模板，不是真正的批量假设生成和挖掘。
- Daily pipeline 已修正为必须显式选择目标因子或组合因子；ExperimentAgent 已具备
  批量实验闭环。
- 已支持基础 train / validation / test 和 walk-forward 样本外切分，但还缺少
  因子衰减、分层稳定性和中性化验证。
- ExperimentAgent / ExperimentStore 已有 MVP、JSONL 历史索引、实验 lineage 记录、
  基础查询能力和自动候选公式执行闭环。
- 缺少多因子组合、因子相关性控制和组合 alpha 构建。
- 缺少面向单只股票的入场、退出、仓位和反证条件。

## 因子准入标准

后续任何因子不能只因为“看起来有道理”就用于实盘观察。一个因子至少要满足：

- 有清晰公式和经济含义。
- 明确数据来源、滞后方式和是否可能未来函数。
- 有足够股票数、日期数和分组样本。
- 样本内和样本外方向一致。
- RankIC / ICIR / 多空收益 / net Sharpe / 最大回撤达到最低门槛。
- 交易成本后仍有统计意义，不是被换手和滑点吃掉。
- 在不同市场阶段、行业或市值分组中不过度依赖单一环境。
- 与已有候选因子不是高度重复。
- CriticAgent 没有给出 `reject_for_now`。

## P0 --- 修正当前研究管线的关键误导点

- [x] 显式支持 daily research 的目标因子选择
  - [x] 当配置多个 `template_ids` 但没有指定 `factor_column` 时，终端和 manifest 必须明确提示不会自动代表“组合因子”
  - [x] 禁止静默使用第一个模板因子作为最终信号，除非配置显式允许
  - [x] 在 manifest 中保存本次真正用于回测、排名和报告的 `selected_factor_column`

- [x] 支持多因子组合
  - [x] 在 FeatureAgent 中支持 `composite_factors`
  - [x] 支持多个基础因子按权重合成
  - [x] 支持按交易日横截面 rank / z-score 后再合成
  - [x] 支持在 daily research 中指定组合因子作为 `factor_column`
  - [x] 在报告中展示组合因子的公式、权重和组件贡献

- [x] 增加因子定义注册表
  - [x] 每个因子保存 name、formula、hypothesis、category、direction、lookback、data_lag
  - [x] 区分模板因子、组合因子、实验生成因子和人工研究因子
  - [x] 报告中明确“这是模板因子 / 组合因子 / 实验候选因子”

## P1 --- 构建 ExperimentAgent：从模板日报升级为因子研究工厂

- [x] 构建 ExperimentAgent
  - [x] 批量生成候选因子实验任务
  - [x] 批量运行 BacktestAgent 评估 factor manifest 中的多个因子
  - [x] 批量调用 CriticAgent 审查结果
  - [x] 保存每个实验的 config、因子定义和结果 artifact
  - [x] 输出实验汇总表
  - [x] 串联 FeatureAgent，从候选因子定义自动生成 factor manifest
  - [x] 支持执行参数化候选公式，而不只是映射到现有可执行模板

- [x] 增加 ExperimentStore
  - [x] 保存单次实验结果到本地 JSON + CSV summary
  - [x] 记录实验 ID、运行时间、请求配置、因子定义和 artifact 路径
  - [x] 保存所有实验结果到本地 JSONL 历史索引
  - [x] 记录 git commit、数据版本、配置 hash
  - [x] 支持按因子类别、股票池、日期范围和 verdict 查询历史实验
  - [x] 支持对失败实验记录失败原因，避免重复浪费时间

- [x] 增加候选因子生成空间
  - [x] 动量类：1/3/5/10/20/60 日收益、相对强弱、趋势斜率
  - [x] 反转类：短期过热回撤、涨跌幅反转、缺口反转
  - [x] 量价类：量能放大、价量背离、成交额突破、换手率变化
  - [x] 波动率类：低波、波动率收缩、振幅变化、回撤修复
  - [x] 流动性类：成交额稳定性、换手率、冲击成本代理
  - [x] 突破类：均线突破、区间新高、回撤后再突破
  - [x] 组合类：多个弱信号加权合成

## P2 --- 样本外和稳健性验证

- [ ] 增加样本外验证
  - [x] 支持 train / validation / test 日期切分
  - [x] 支持 walk-forward validation
  - [x] 对比样本内和样本外 IC、RankIC、Sharpe、回撤、换手、成本后收益
  - [x] 在报告中明确标记样本外是否通过

- [ ] 增加因子衰减测试
  - [ ] 测试 forward return horizon：1D、3D、5D、10D、20D
  - [ ] 输出 IC decay curve
  - [ ] 判断因子适合日频观察、短线交易还是中期持有

- [ ] 增加稳健性和分层评估
  - [ ] 按年份 / 牛熊震荡阶段拆分表现
  - [ ] 按行业或主题股票池拆分表现
  - [ ] 按市值、成交额、波动率分组检查稳定性
  - [ ] 输出“因子在哪些环境有效 / 失效”

- [ ] 增加数据泄漏和偏差检查
  - [ ] 检查因子计算是否使用未来价格或未来成交量
  - [ ] 记录 universe 是否 point-in-time
  - [ ] 对静态股票池输出幸存者偏差提示
  - [ ] 对停牌、涨跌停、新股、ST 和退市风险处理写入报告

## P3 --- 因子组合和 Alpha 选择

- [ ] 增加因子相关性分析
  - [ ] 计算候选因子之间的横截面相关性
  - [ ] 识别高度重复因子
  - [ ] 避免把多个表达同一逻辑的因子重复加权

- [ ] 增加多因子组合研究
  - [ ] 支持等权组合
  - [ ] 支持人工指定权重组合
  - [ ] 支持简单网格搜索权重
  - [ ] 支持按样本外表现选择组合
  - [ ] 比较单因子和组合因子的样本外表现

- [ ] 增加 Alpha 候选池
  - [ ] 只有通过样本外和 CriticAgent 的因子才能进入候选池
  - [ ] 保存每个 alpha 的生命周期：created、watching、promoted、retired、rejected
  - [ ] 对失效 alpha 做退休和复盘

## P4 --- 股票观察和交易前决策

- [ ] 构建 DecisionAgent
  - [ ] 输入 daily ranking、交易约束、回测质量、风险指标和最近走势
  - [ ] 输出 `观察` / `可小仓试错` / `暂不介入` / `考虑退出`
  - [ ] 为每个结论写明核心理由和反证条件
  - [ ] 对银轮股份这类关注股生成单票决策摘要
  - [ ] 明确提示“不是投资建议，需要人工确认”

- [ ] 增加入场规则研究
  - [ ] 连续 N 天进入 Top 区间
  - [ ] 因子分数从负转正或持续改善
  - [ ] 趋势过滤：均线、突破、回撤后修复
  - [ ] 风险过滤：涨停、过热、波动过大、流动性不足
  - [ ] 输出候选入场触发条件和信号失效条件

- [ ] 增加卖出规则研究
  - [ ] 跌出排名 Top 区间连续 N 天
  - [ ] 因子分数转弱或反转
  - [ ] 最大回撤或波动率超过阈值
  - [ ] 触发止损、止盈或持仓天数上限
  - [ ] 输出卖出原因和继续持有的反证条件

- [ ] 增加单票观察报告
  - [ ] 当前因子暴露
  - [ ] 同行相对位置
  - [ ] 近期趋势和波动
  - [ ] 关键风险
  - [ ] 下一交易日观察条件

## P5 --- 组合、仓位和风险控制

- [ ] 构建 PortfolioAgent
  - [ ] 将候选排名转换为 watchlist 或 paper portfolio
  - [ ] 增加简单仓位规则
  - [ ] 增加最大持仓数量和单票最大暴露
  - [ ] 增加现金和风险约束
  - [ ] 输出“如果买，最多买多少”的研究辅助结果

- [ ] 增加组合风险检查
  - [ ] 行业集中度
  - [ ] 单票集中度
  - [ ] 高相关持仓聚集
  - [ ] 最大回撤预算
  - [ ] 流动性和换手约束

- [ ] 增加 paper trading log
  - [ ] 保存计划交易
  - [ ] 手动保存实际成交
  - [ ] 对比预期收益和实际收益
  - [ ] 每周复盘错误
  - [ ] 将交易结果反馈给因子和 DecisionAgent

## P6 --- 数据升级

- [ ] 增加行业和主题数据
  - [ ] 保存每只股票的行业分类
  - [ ] 支持汽车零部件、新能源车、机器人等主题 watchlist
  - [ ] 对银轮股份这类关注股自动生成可比股票池

- [ ] 增加基础面和估值数据
  - [ ] PE、PB、PS、ROE、营收增速、利润增速
  - [ ] 记录财报发布日期，避免未来函数
  - [ ] 支持基础面因子和量价因子结合

- [ ] 增加指数和市场状态数据
  - [ ] 大盘趋势
  - [ ] 行业指数趋势
  - [ ] 市场宽度
  - [ ] 风格状态：大盘 / 小盘、成长 / 价值

## P7 --- 研究记忆和复盘纪律

- [ ] 增强 MemoryAgent
  - [ ] 保存被拒绝因子的失败原因
  - [ ] 保存通过因子的后续表现
  - [ ] 对相似因子给出历史提醒
  - [ ] 自动生成每周研究复盘

- [ ] 增加人工确认记录
  - [ ] 记录是否真的采用 Agent 输出
  - [ ] 记录人工否决原因
  - [ ] 记录实际交易后的复盘结果
  - [ ] 区分“因子错了”和“人工没有遵守规则”

- [ ] 增加每日研究 checklist
  - [ ] 数据更新完成
  - [ ] 失败股票已复核
  - [ ] 候选排名已生成
  - [ ] 风险报告已阅读
  - [ ] 人工决策已记录

## P8 --- Dashboard 和自用工作流

- [ ] 改进 dashboard
  - [ ] 增加实验结果列表
  - [ ] 增加样本外表现过滤
  - [ ] 增加 benchmark status / critic verdict 过滤
  - [ ] 增加因子类别过滤
  - [ ] 直接展示每日候选股排名和单票决策摘要

- [ ] 增加配置模板
  - [ ] 创建 `configs/daily_research.example.yaml`
  - [ ] 创建 `configs/experiment.example.yaml`
  - [ ] 创建 `configs/cost_profile.example.yaml`
  - [ ] 创建 `configs/watchlist.example.yaml`
  - [ ] 创建 `configs/yinlun_watchlist.example.yaml`

- [ ] 增加 watchlist 支持
  - [ ] 读取 `configs/watchlist.yaml`
  - [ ] 支持个人关注股票池
  - [ ] 支持 watchlist 专属报告
  - [ ] 支持多个 watchlist 批量运行

## 已完成基础能力

- [x] 稳定真实 AkShare 数据接入
- [x] 增加 daily research pipeline
- [x] 支持中英双语输出
- [x] 生成每日候选股排名
- [x] 增加 A 股交易约束
- [x] 增加交易成本真实性
- [x] 收紧默认回测质量门槛
- [x] 构建 CriticAgent
- [x] 保存研究记忆和报告
- [x] 增加因子定义注册表
- [x] 提供 dashboard 复盘入口

## 建议下一步

优先继续完成 P2，而不是继续堆更多简单模板因子：

1. 扩展样本外验证：加入因子衰减测试。
2. 增加因子相关性分析，避免重复研究高度相似信号。
3. 在有了通过验证的候选 alpha 后，再构建 DecisionAgent 来服务银轮股份这类单票观察。

这条路线更接近量化公司 Quant Researcher 的工作方式：先大规模研究和验证因子，
再讨论具体股票是否值得观察、何时入场、何时退出。
