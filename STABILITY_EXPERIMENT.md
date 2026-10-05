# 多种子、字段顺序与 Markov 基线实验

## 固定设计

在扩大口径数据上，保留原来的分箱、日期/股票划分、模型主体、优化器和 1,000 步预算。使用 seed 42、43、44，比较：

| 设置 | 网络 | 字段顺序 |
|---|---|---|
| joint | 联合 token Transformer | 不适用；编码仍使用固定的三个字段索引 |
| seq_tpv | 逐字段 Transformer | 间隔→价格→成交量 |
| seq_ptv | 逐字段 Transformer | 价格→间隔→成交量 |

每模型 32,000 次目标事件曝光，历史为 128 个事件。同 seed 的神经网络设置使用相同目标索引流。seed 42 的 joint 和 seq_tpv 沿用上一轮检查点，不重新训练或覆盖原报告。七个新增模型分开保存。

这是相同事件预算和 Transformer 主体结构下的对照，不是等计算量对照。joint 输入长度为 128 token，逐字段输入最长 386 token；两者的总参数量分别为 2,001,024 和 1,863,168，主体参数量均为 1,779,840。不能仅凭 NLL 较低就断言某种编码计算效率更高。

另加入三个统计基线：

- frequency_full：全部合格训练目标的频率分布。
- markov_full：使用全部合格训练目标，按前一事件的联合类别估计下一事件分布。
- markov_matched：每个 seed 只使用对应神经网络实际抽到的 32,000 次目标事件及各自前一事件，重复曝光保留计数。

Markov 转移只在同股票同交易日内构造，不连接两个 session。用训练频率作为 Dirichlet 先验，未见过的上一事件类别回退到该先验。先验强度的预设网格为 1、10、100、1000，仅按验证集 NLL 选择；两个测试集均不参与选择。

## 执行

在项目目录安装并激活 Python 环境后：

```bash
python -m taq_lab.study run --plan configs/stability-study.json --workers 2
```

完成过的训练自动复用，未完成的训练从检查点继续。整个计划、各配置及数据指纹必须保持一致，否则拒绝混用。修改研究设计应使用新计划和输出目录。

分开执行也可以：

```bash
python -m taq_lab.study train --plan configs/stability-study.json --workers 2
python -m taq_lab.study status --plan configs/stability-study.json
python -m taq_lab.study evaluate --plan configs/stability-study.json
```

本机使用两个独立 CPU 训练进程，每个四线程；不会在共享全局随机状态的 Python 线程里训练模型。并发训练的耗时受资源竞争影响，不能与之前独占 CPU 的时长直接对比。两个进程只影响调度，不修改 batch size、目标事件预算或模型配置。

训练日志保存在 runs/stability_study/logs/，总体状态为 progress.json。新检查点位于 runs/stability_seed43、stability_seed44、stability_order_ptv_seed42/43/44。旧 seed42 结果仍在 inclusive_seed42。

## 输出与判断

所有九个模型训练完成后统一评价：每个 split 使用相同的 3,000 个窗口，选检查点只依据验证 NLL。输出 runs/stability_study/report.html、report.md、metrics.csv、session_metrics.csv、paired_seed_differences.csv、training_runs.csv、baseline_parameters.json 及逐事件分数。

主要比较同 seed 配对差值：seq_tpv − joint、seq_ptv − seq_tpv、seq_tpv − markov_matched。差值为负说明左边损失更低。固定全量 Markov 同时作为充分使用现有训练数据的强基线，不能假装它与神经网络使用相同数量的监督目标。

均值 ± 标准差只反映随机种子波动，不是市场泛化置信区间；每个窗口高度重叠，测试日期只有两天。上一轮已看过 seed42 的测试结果，后续确认性研究仍需新日期作为最终保留集。本轮不按测试结果调整架构、预算、字段顺序或平滑网格。

这一步保持最后事件监督目标不变；密集监督、下游微调、更多历史和 GPU 扩展是之后独立的实验，不混入本次对照。
