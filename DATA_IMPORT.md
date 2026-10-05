# 真实 TAQ 导入与试跑记录

来源：i8zqtj4euooaohof_csv.zip。四只股票 AAPL、MSFT、AMZN、NVDA，2025-02-03 至 2025-02-14，共 10 个交易日、40 个股票/日期组合，全部齐全。

原始记录：31,738,845；保留成交：2,631,828；事件：2,631,796。每个 session 的首笔成交没有前序间隔，因此少一个特征事件。

| 用途 | 股票/日期组合数 | 事件数 |
|---|---:|---:|
| train | 18 | 1,246,741 |
| val | 6 | 340,651 |
| test | 6 | 353,960 |
| heldout | 2 | 690,444 |

NVDA 仅使用最后两个交易日作未见股票测试，前八天按实验设计排除。

过滤计数按顺序统计，互不重复；被排除的成交条件不等于错误数据。当前严格 regular-sale 子样本需在正式研究前审查。

- invalid_timestamp: 0
- invalid_price_or_size: 6
- excluded_correction: 1,306
- excluded_condition: 25,662,625
- outside_session: 68
- outside_fixed_split: 3,443,012

## CPU 流程验证

两种模型各训练 10 步、batch size 4，每模型仅 40 次目标事件曝光；各评估集最多抽样 32 个窗口。此结果只能证明流程跑通，不能比较模型优劣或证明交易能力。

- joint: 2,001,024 参数，训练阶段 0.44 秒；CPU，未用 GPU。
- sequential: 1,863,168 参数，训练阶段 1.26 秒；CPU，未用 GPU。

## 后续运行

正式训练使用 configs/taq-import.json，输出 runs/main；短试跑使用 configs/taq-smoke.json，输出 runs/real_smoke。两者共用已处理数据，输出分开。

```bash
python -m taq_lab run --config configs/taq-import.json
```

已有处理结果不用再次 prepare。换数据、清洗规则或分箱设置时需要重新处理；不要将旧 AAPL 单日文件叠加进本次导出。
