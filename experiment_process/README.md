# 实验处理入口

全部命令、数据输出结构及环境见项目根README。这里不执行绘图，也不调用old_experiment_code。

| 文件 | 用途/主要入口 |
|---|---|
| generate_inductive_analysis_cache.py | 主图2/3归纳损失和潜在统计；ensure_inductive_caches |
| generate_transductive_analysis_cache.py | 补充直推式损失；ensure_transductive_caches |
| generate_model_parameter_analysis_cache.py | 损失/训练参数敏感性、缺失基础PRISM补训；ensure_figure/ensure_prism_checkpoint |
| generate_resilience_inference_cache.py | SIS二分类、Neuronal二/三分类；严格加载、缺失MLP补训、KNN参考库、统计；ensure_caches |
| resinf_inference.py | 共享ResInf严格加载、拓扑预处理、缺失/空权重补训 |
| generate_figure5_tsne_cache.py | 潜在表征t-SNE；ensure_figure5_caches |
| run_full_inference_safe.py | 原监督全量流程串行启动、内存保护；本轮不运行 |
| benchmark_sis_scaling.py | 唯一规模处理入口：read_config/preflight/prepare_models/run/aggregate；含隔离smoke_test |
| sis_scaling.example.yaml | 规模实验数据、比例、10次重复和计时配置 |
| test_resilience_inference_smoke.py | 9项隔离回归测试，不写正式结果 |

规模处理仅summary.csv+records.json（历史替换进入archive）；逐组/方法/重复保存，可续接未完成项。原监督实验保留旧抽样、池化和最小—最大误差条协议；规模实验单独采用共享完整测试集、逐模型weighted F1及ddof=0。不要混用两种协议。

2026-09-07：原9项隔离测试通过；规模微型测试覆盖已有MLP/ResInf输出、归一化等价、KNN固定参考库、GBB边界、缓存命中/汇总恢复/断点与三格式图。九类原图另通过缓存导出21个PNG/PDF/SVG到临时目录，原图哈希未变。测试数据非论文结果；没有补训正式模型。
