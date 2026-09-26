# 绘图入口

环境、完整命令和输出名称见项目根README。优先缓存；缺缓存调用experiment_process，必要时由处理层加载/补训模型。不调用old_experiment_code。

| 文件 | 实验 |
|---|---|
| plot_inductive_loss_task_analysis.py | 主图2归纳损失 |
| plot_inductive_representation_early_warning.py | 主图3潜在统计 |
| plot_resilience_inference_performance.py | 主图4监督性能 |
| plot_latent_embedding_tsne.py | 主图5潜在表示 |
| plot_transductive_loss_task_analysis.py | 直推损失 |
| plot_loss_parameter_sensitivity.py | 重构损失参数 |
| plot_training_parameter_sensitivity.py | 学习率/dropout |
| plot_knn_gbb_comparison.py | KNN–GBB对比 |
| plot_neuronal_binary_classification.py | Neuronal二分类 |
| plot_sis_scaling.py | SIS规模：时间、CPU/GPU内存、weighted F1；SVG/PDF/PNG |
| resilience_inference_plotting.py | 三种监督绘图共享加载/样式 |
| model_parameter_plotting.py | 两种参数敏感性图共享加载/样式 |

前九个入口支持--device、--force-cache、--output-dir，监督图另支持--ratios/--trials。
规模入口使用--config、--results、--output、--force-cache；不完整结果默认不绘图，可显式--allow-partial并显示标记。
原图布局与误差条不变；监督比较在每项任务内使用相同测试样本 ID：SIS 与神经元三分类使用全部有效的 N>20 样本，神经元二分类使用共同的平衡有效子集。规模图为逐模型均值±标准差(ddof=0)，GBB不伪造10次模型重复。CPU RSS和GPU峰值分别绘制，不相加。仅临时TEST ONLY图验证导出，不改原论文图片。
