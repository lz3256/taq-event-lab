# TAQ Event Lab

最新阶段：[时钟任务与冻结表征结果](CLOCK_PROBE_RESULTS.md) · 离线报告 · [预先固定的实验方案](CLOCK_PROBE_PROTOCOL.md)。

基于 TAQ 成交序列，从零训练小型 Transformer，比较 **联合事件 token** 与 **逐字段 token**。包含真实数据导入、WRDS 下载入口、清洗审计、训练集分箱、两种模型、频率基线、断点恢复、未见股票测试、事件生成、CSV 指标、HTML 报告和自动测试。

这是 **TradeFM / LOBS5 思路的小规模改编和受控实验**，不是两篇论文的官方复现。当前包含成交事件预训练和下游方向分类微调，不包含报价模型、完整订单簿、做市模拟或成本回测。生成结果和预测损失不能证明盈利能力。

最新已完成 **密集监督三种子复现与 CPU 等训练时间实验**：相同目标曝光下逐字段领先；本次单种子等时间实验中联合 token 在两个测试集反超。优先查看 [结果与简历表述](EFFICIENCY_RESULTS.md) 和 汇总报告。

## 1. 安装

Python 3.10+。在本目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows 使用 `.venv\Scripts\activate`。Linux CUDA 用户先按 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 PyTorch。`device=auto` 优先 CUDA，其次 Apple MPS，再次 CPU。正式训练前先测吞吐量。

`requirements-tested.txt` 记录交付时实际测试的 Python 包版本（macOS arm64 / Python 3.12）；它不是所有平台通用的 CUDA 锁文件。

## 2. 一条命令跑通完整流程

```bash
python -m taq_lab demo --directory runs/demo --steps 18
```

目录必须是新的，避免覆盖结果。此命令生成明确标记的**合成数据**，在 CPU 上训练两种小模型，评价频率基线和模型，生成 10 个事件，并输出：

```text
runs/demo/
  config.json
  raw/synthetic_trades.csv
  raw/SYNTHETIC_DATA.json
  prepared/manifest.json
  prepared/audit.json
  results/report.html
  results/report.md
  results/summary.csv
  results/metrics.csv
  results/session_metrics.csv
  results/validation_by_events.png
  results/validation_by_time.png
  results/joint/{best.pt,last.pt,history.jsonl,...}
  results/sequential/{best.pt,last.pt,history.jsonl,...}
```

交付中 `examples/synthetic_demo/` 是已经运行过的演示。**所有演示指标只证明流程可运行，不是市场研究结果。** 自己重跑请使用新的目录。

## 3. 接入真实 TAQ

### 方式 A：现有 CSV / Parquet / CSV ZIP 导出

将数据放到 `data/raw/`，修改 `configs/taq.json` 中的 `input_glob`、股票、交易日期。所有配置中的路径都**相对于配置文件所在目录**解析，而不是当前工作目录。

本次四股票网页导出的配置为 `configs/taq-import.json`，直接读取 `data/raw/i8zqtj4euooaohof_csv.zip`，无需手动解压。运行 `python -m taq_lab prepare --config configs/taq-import.json` 进行清洗；后续训练、评估也使用同一配置。不要同时加入之前 AAPL 单日导出，否则会产生重叠记录。

本机已完成该文件的清洗及真实数据 CPU 短试跑，详情见 `DATA_IMPORT.md`。`configs/taq-smoke.json` 仅跑每模型 10 步、每个评估集最多 32 个窗口，输出 `runs/real_smoke/report.html`，仅验证流程。正式训练运行 `python -m taq_lab run --config configs/taq-import.json`，输出独立的 `runs/main/`。项目 ZIP 不含真实原始数据、处理后的数据或真实数据模型权重，换机器需自行复制数据并重新 prepare。

支持两种输入形态（列名大小写不敏感）：

| 标准列 | WRDS 对应列 | 含义 |
|---|---|---|
| `symbol` | `sym_root`, `sym_suffix` | 股票；带后缀会组合成 `ROOT.SUFFIX`，不会混合不同 share class |
| `timestamp` | `date` + `time_m` | ISO 日期时间，或日期与时刻分列 |
| `price` | `price` | 实际美元价格，不是未经换算的定点整数 |
| `size` | `size` | 成交股数 |
| `tr_corr` | `tr_corr` | 修正码 |
| `tr_scond` | `tr_scond` | 成交条件码，保留原始值 |

标准 CSV 例子：

```csv
symbol,timestamp,price,size,tr_corr,tr_scond
AAPL,2025-02-03T09:30:00.000000001,230.10,100,00,@
AAPL,2025-02-03T09:30:00.100000001,230.11,200,00,@
```

无时区时间按 `data.timezone` 解释（默认纽约），有时区时间转换到该时区。浮点 epoch 时间戳不接受，以免静默丢失精度。保留纳秒直到计算时间差。

真实数据配置 `configs/taq.json` 为 2025-02-03 至 2025-02-14 的 10 个交易日：AAPL、MSFT、AMZN 用于训练及按日期划分的验证/测试，NVDA 仅用于跨股票测试，暂不包含 META。合成演示仍使用原有五只股票。请先核实自己的访问日期和股票数据是否齐全。缺失/过短的 session 默认报错；若研究上允许缺失，可显式设 `missing_policy="skip"`，审计文件会记录。

### 方式 B：WRDS 只读下载

```bash
python -m pip install -e ".[wrds]"
python -m taq_lab download-wrds --config configs/taq.json --output data/raw --username YOUR_WRDS_USERNAME
```

凭据交给 WRDS 库在本机交互处理，程序不接受密码 CLI 参数，不包含任何账号。默认访问 `taqmsec.ctm_YYYYMMDD`，通过参数绑定传递股票和时段，分块读取、逐日保存 Parquet。不同账号库名可能不同，可用 `--library` 指定。已存在导出会被拒绝覆盖。

**WRDS 的真实连接与账号权限未在交付环境验证。** 下载代码按 WRDS 官方 Python API 编写并用模拟连接测试；实际权限、表名、字段和节假日由账号数据决定。该适配器当前只下载无 suffix 的股票。也可以直接使用网页导出文件。

### 清洗边界

- 默认只保留 `tr_corr` 为 `0` / `00`、完整 `tr_scond` 去首尾空格后为 `""` / `"@"` 的记录。这是**保守的 regular-sale 子样本**，不是通用 TAQ 清洗标准；会排除一些正常但带附加条件的成交。
- 根据你所使用年份的数据字典审查 `allowed_sale_conditions`。不自动忽略内部字符，不把 TAQ 成交取消/修正当成订单撤单。
- 保留纽约时间 `[09:30:00,16:00:00)`，排除非法时间、非正价格或数量，按固定日期/股票列表筛选。默认时段不自动识别半日市；半日市请分配置处理或排除。
- 时间相同的记录保留稳定输入文件/行顺序，并记录数量；这不是推断出的交易所全局因果顺序。没有可信交易 ID 时不去重，以免删除真实的相同成交。**输入导出不能互相重叠。**
- 清洗后才计算三个字段：`log1p(成交间隔秒)`、`10000 * log(当前成交价 / 前一成交价)`、`log1p(成交股数)`。每个股票/交易日丢弃第一条记录，不跨日计算。
- 只在训练 session 上拟合分位数边界。默认每字段 8 档；重复分位点允许存在，可能导致空档；极端值进入边界档，超出训练范围的比例写入 manifest。反解只是档位代表值，不是原始数据的无损恢复。
- prepare 按 25 万行分块读取 CSV、CSV ZIP 和 Parquet，保留清洗后的行并进行全局排序，保证跨块同时间记录的稳定顺序。原始数据无需一次载入，但全部保留记录及特征仍需适配内存；**不是十亿事件级管道**。审计中记录原始股票/日期覆盖及过滤计数。

## 4. 真实数据完整运行

本次第一轮正式对照使用 `configs/taq-inclusive.json`：在严格口径上纳入带 `I`/`F` 标记的普通成交，独立生成 `data/prepared_inclusive/`，两种模型各训练 1,000 步，结果写到 `runs/inclusive_seed42/`。研究口径、训练预算和结果解释见 `FIRST_EXPERIMENT.md`。

该轮已在本机 CPU 完成，中文结果见 `FIRST_RESULTS.md`，HTML 报告为 `runs/inclusive_seed42/report.html`。两模型均优于频率基线；逐字段模型在此次未来日期和 NVDA 测试上的 NLL 更低，但训练耗时约为联合模型 6 倍。这是单 seed、单字段顺序结果，不能推断普遍优势或交易盈利。

```bash
python -m taq_lab run --config configs/taq-inclusive.json
# 中断后保持配置不变，使用：
python -m taq_lab run --config configs/taq-inclusive.json --resume
```

下面是通用模板命令；网页 ZIP 数据使用上面的 inclusive 配置，或使用 `taq-import.json` 重做严格口径实验。

```bash
python -m taq_lab run --config configs/taq.json
```

也可逐步运行：

```bash
python -m taq_lab prepare --config configs/taq.json
python -m taq_lab train --config configs/taq.json --model joint
python -m taq_lab train --config configs/taq.json --model sequential
python -m taq_lab evaluate --config configs/taq.json
python -m taq_lab report --config configs/taq.json
```

输出到配置的 `output_dir`（默认 `runs/main/`）。生成报告后直接打开 `report.html`。报告图是同一轮训练的事件预算和实际训练时间曲线，并不冒充严格的多预算 scaling-law 实验。

### 先做短试跑，再决定预算

```bash
python -m taq_lab train --config configs/taq.json --model joint --stop-after 20
python -m taq_lab train --config configs/taq.json --model sequential --stop-after 20
```

`--stop-after` 是绝对优化步数，保持原 `max_steps` 对应的学习率计划。查看各模型的 `training_summary.json`。吞吐量是**被监督的完整目标事件/秒**，包含取样、数据传输和优化，排除验证；另记录历史事件处理次数。它不是 token/s，也不是唯一事件数。

恢复时保持配置不变：

```bash
python -m taq_lab train --config configs/taq.json --model joint --resume
python -m taq_lab train --config configs/taq.json --model sequential --resume
python -m taq_lab evaluate --config configs/taq.json
python -m taq_lab report --config configs/taq.json
```

或使用 `run --resume`：已有 checkpoint 的模型恢复，尚未开始的模型从头训练。checkpoint 保存模型、优化器、取样 RNG、CPU/CUDA/MPS RNG、进度与学习曲线。数据哈希或配置变化会拒绝恢复；新增实验应改输出目录。每个 log/eval 点保存 `last.pt`，验证改善时保存 `best.pt`。突然中断只会丢失最后保存点之后的工作。

`prepare --overwrite` 显式重新生成处理后数据；旧训练结果不会自动覆盖，也不会与新数据静默混用。

### 可选的离散事件生成

```bash
python -m taq_lab generate --config configs/taq.json --model sequential --steps 20 --sample
```

以 heldout 中第一个窗口为种子生成未来成交事件，仅滚动使用生成值；不会喂真实未来字段。CSV 中连续值是 bin center 近似，附带 metadata。没有撮合引擎，也不代表生成了可执行订单。

## 5. 实验口径

### 多种子与字段顺序实验

第二轮使用 seed 42/43/44，对比联合 token、间隔→价格→成交量、价格→间隔→成交量，复用已有 seed42 的两组检查点，另训练七组。同时加入全训练数据的一阶 Markov 基线，以及匹配每个 seed 的 32,000 次神经网络目标曝光的 Markov 基线。详见 `STABILITY_EXPERIMENT.md`。

```bash
python -m taq_lab.study run --plan configs/stability-study.json --workers 2
python -m taq_lab.study status --plan configs/stability-study.json
```

同一命令自动复用完成项、恢复中断项。结果在 `runs/stability_study/`，保留每个 seed、每个股票/日期和每个评价窗口的分数。均值 ± 标准差描述随机种子波动，不能当成市场泛化置信区间。

本轮九组训练和统一评估已完成，实测汇总见 `STABILITY_RESULTS.md`。TPV 逐字段模型在三个种子的两个测试集上均优于联合 token 和全量 Markov；改用 PTV 顺序未出现稳定提升。联合 token 未超过全量 Markov，说明仅与频率基线对比不足以确认优势。结果仍受有限日期、训练预算和交易数据范围限制。

### 密集监督修正

此前的 1,000 步实验每窗口只监督末事件，共 32,000 次目标曝光。新增 `train.supervision="dense"` 对窗口中的全部下一事件位置计算损失；每窗口 128 个目标，同步数为 4,096,000 次曝光。评估仍只看完整历史之后的末事件 NLL。独特目标数另外统计，不把重复曝光当作独立数据。

```bash
python -m taq_lab run --config configs/taq-dense-seed42.json
python -m taq_lab.dense_report --plan configs/dense-pilot.json
```

设计、损失对齐、兼容性与曲线解释见 `DENSE_EXPERIMENT.md`。新目录 `runs/dense_seed42/`、`runs/dense_pilot/` 保留旧结果。密集模式的前部位置历史较短，计算量也不完全相同，所以相同步数实验不等于只改变独立样本数的对照。

首轮 seed42 两组训练及统一评价已完成，见 [DENSE_RESULTS.md](DENSE_RESULTS.md)：监督曝光从每模型 32,000 增加至 4,096,000，两种模型均改善；逐字段优势在同股票未来日期从 0.2034 缩小至 0.0462 nats/event。该结果是单种子诊断，尚未重跑下游迁移。

多种子与等训练时间扩展的固定方案见 [COMPUTE_EXPERIMENT.md](COMPUTE_EXPERIMENT.md)。三种子实验使用 `configs/dense-stability.json`；独立的 CPU 时间预算实验使用 `configs/time-budget-pilot.json`。时间模式按累计训练秒数调度学习率和验证，不能用固定步数研究模块代替。

```bash
python -m taq_lab.study run --plan configs/dense-stability.json --workers 1
python -m taq_lab.compute_study run --plan configs/time-budget-pilot.json
python -m taq_lab.followup_report --sparse runs/stability_study --dense runs/dense_stability --timed runs/time_budget_comparison --output runs/dense_followup
```

上述命令依次运行，以避免本项目内的训练资源竞争。最后一条命令要求旧三种子研究和两组新实验均已生成完整评价明细。

1. **同一信息集**：两模型均使用前 `context_events` 个完整事件（默认 128）。联合模型看 128 个 token，逐字段模型看 384 个历史 token，并在预测后续字段时看到当前事件已知前缀。
2. **同一预测目标**：旧配置监督末事件，新密集配置监督每个下一事件；两种编码在相同监督模式下对齐目标位置。联合模型预测 512 类；逐字段模型顺序预测三个 8 类分布，正确屏蔽不属于当前字段的 token。三项交叉熵**求和**，得到 nats/event，再对 batch 和监督事件取平均。
3. **同一训练取样流**：独立 seeded sampler 有放回抽取相同的窗口索引。相同 seed/batch/steps 的两种模型看到相同目标事件。模型初始化形状不同，不要求初始权重完全相同。
4. **报告参数差异**：Transformer 层/宽度/头数相同，词表与位置表导致总参数不同。报告 total 和 backbone 参数。默认大配置约 1–5M 参数，demo 使用更小模型以便快速验通。
5. **固定切分**：seen 股票前 6 天 train、中间 2 天 val、最后 2 天 test；heldout 股票只使用最后 2 天测试。训练/分箱从未看 heldout。窗口不跨 session。
6. **验证集选模型**：`best.pt` 只由 val 决定。val/test/heldout 在三种方法间使用完全相同、确定性的窗口索引；超过 `max_windows` 时均匀子采样。逐事件 CSV 可以核对。频率基线对所有**可用训练目标**计数并做加一平滑。
7. **一个 seed 是初步结果**：报告 event-weighted 和 session-macro NLL，保留逐日/股票结果。不对重叠窗口使用 IID 显著性假设。正式研究应多 seed、更多日期、按日/股票评估不确定性。
8. **生成延迟**：batch=1、设备同步、固定历史、无 KV cache 的完整事件贪心生成。逐字段模型重复计算前缀，因此这个实现的延迟结果不代表优化后的理论极限。不是交易所端到端延迟。
9. **训练精度**：默认 fp32。支持 CUDA bf16，需设 `train.precision="bf16"` 且硬件支持。不伪装已经测试 CUDA、多卡或 bf16；交付实际验证范围见 `VALIDATION.md`。

## 6. 下游方向预测：预训练 vs 随机初始化

使用 TPV 编码器比较少量标注下的迁移效果：512/2,048 个标注样本 × 三个种子 × 两种初始化，共 12 组神经网络训练，另有类别频率与历史特征逻辑回归基线。

```bash
python -m taq_lab.downstream run --plan configs/downstream-study.json
```

任务、固定预算、标签边界与复现说明见 `DOWNSTREAM_EXPERIMENT.md`。输出 `runs/downstream_study/report.html`；这是三分类 NLL，不与事件生成 NLL 比较。继续使用现有日期，所以是探索性实验；新日期仍需作为独立最终验证。

本轮 12 组训练和评估已完成，见 `DOWNSTREAM_RESULTS.md`。当前未观察到稳定的预训练迁移收益；神经网络的预测几乎全是平稳类，不能把约 81% 的准确率解读成识别涨跌的能力。2,048 标注样本下，逻辑回归在同股票未来日期的 NLL 与平衡准确率优于神经网络。

上述迁移结果使用旧末事件监督的编码器，不代表密集预训练的迁移表现。密集监督修正及其事件预测结果见 `DENSE_RESULTS.md`；新的标签和线性探针尚未实现。

## 7. 换字段顺序 / 随机种子

复制 `configs/taq.json`，保持同目录，修改：

```json
"tokenizer": {"bins": 8, "field_order": [1, 2, 0]}
```

`0=时间间隔，1=价格变化，2=成交量`。同时改 `output_dir`；改变 seed 也要使用新输出目录。字段顺序不改变 prepared 数据，因此可以复用同一 `prepared_dir`。为维持严格恢复检查，每个配置的运行独立保存。

## 8. 测试

```bash
python -m pytest -q
```

关键测试覆盖：编码双射、分箱无测试集泄漏、纳秒/时区处理、条件码清洗、股票后缀、切分与窗口边界、因果 mask、三种字段顺序的联合 NLL、极小样本学习、生成范围、数据哈希、断点恢复与连续训练的权重一致性，以及完整报告流程。另验证 Markov 不跨 session、未见状态回退、匹配训练目标流、多种子实验配置一致性和统一评价窗口。

## 时钟任务与冻结表征（阶段一）

新研究独立保留在 `runs/clock_probes`；不覆盖旧的 20 笔成交方向实验。完整方案见 [CLOCK_PROBE_PROTOCOL.md](CLOCK_PROBE_PROTOCOL.md)。所有既有评价日期都按开发数据处理。

```bash
python -m taq_lab.clock_probes run --plan configs/clock-probes.json
# 可分阶段运行或恢复：prepare / fields / probes / report
python -m taq_lab.clock_audit --root runs/clock_probes
```

前提是原始 TAQ 文件、`prepared_inclusive` 及三个密集监督种子的 checkpoint 均在配置指定位置。首次构造任务会重新读取原始文件，校验 SHA256，并将重建 token 与已有预训练事件逐项核对。需要精确原始时间戳、价格、数量，不能仅凭旧 token 还原标签。

输出包含时钟标签与类别审计、三项上游条件 NLL、标注效率曲线、配对种子差值、按股票日期统计和离线 HTML。冻结编码器在 eval 模式下缓存表征；各组分别按调参日期选择 L2；同种子共享嵌套标注子集。模型拟合标签预算不包括共享的标签阈值校准和验证标签。

这一阶段不包含端到端微调或新日期的最终确认。配置或相关代码改变时，缓存校验会拒绝混用，需使用新的研究输出目录。

## 9. 项目文件

```text
taq_lab/
  common.py          配置、哈希、原子 JSON 写入
  data.py            TAQ 导入/审计/切分、窗口、合成数据
  tokenization.py    训练集分箱与两种编码
  model.py           因果 Transformer、事件损失、生成
  engine.py          训练/恢复、基线、评价、性能测量
  report.py          图表、CSV 汇总、HTML/Markdown 报告
  baselines.py       一阶 Markov 计数、平滑和验证集选参
  study.py           固定多种子实验计划、复用与恢复
  study_report.py    匹配曝光基线、统一评价与种子统计
  dense_report.py    密集/末事件监督、目标覆盖与学习曲线比较
  compute_study.py   串行 CPU 等训练时间实验及预算审计
  followup_report.py 三种子监督对照与单种子时间实验的汇总
  downstream_data.py 未来方向标签、因果窗口、固定标注子集
  downstream.py     预训练/随机初始化微调、分类与逻辑回归基线
  downstream_report.py 下游分类指标、配对差值与报告
  clock_data.py      原始时间戳对齐、三项时钟标签与共享因果特征
  nll_fields.py      联合/逐字段/Markov 的可比条件损失拆解
  clock_probes.py    冻结表征缓存、配对标注预算与正则选择
  clock_report.py    任务审计、配对差值和离线开发报告
  clock_audit.py     保存参数重建预测、标注子集和旧 NLL 独立核验
  wrds_download.py   可选 WRDS 下载器
  cli.py             CLI 与完整 demo
configs/taq.json      真实数据实验配置
tests/               自动验证
examples/            已跑通的合成演示及结果
```

数据和研究结果不应混进代码仓库。`.gitignore` 排除了常用数据/运行目录和权重；发布自己的结果前审查数据授权，示例仅包含本项目生成的合成数据。

## References

- [TradeFM](https://arxiv.org/abs/2602.23784)：联合事件表示及自回归预训练思路。
- [LOBS5](https://arxiv.org/abs/2309.00638)：逐字段/子字段生成思路；本项目没有复刻其 S5 和订单簿分支。
- [NYSE Daily TAQ](https://www.nyse.com/data-products/catalog/daily-taq)：产品范围。
- [WRDS Python API](https://github.com/wharton/wrds/blob/main/wrds/sql.py)：`raw_sql` 参数与分块访问。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：因果注意力实现。
