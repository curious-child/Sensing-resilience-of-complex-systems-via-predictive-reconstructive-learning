# SIS统一数据生成

在项目根目录运行：
```bash
python dataset/sis_dataset_generation/generate_sis_dataset.py --config dataset/sis_dataset_generation/config.yaml
# 临时验证，不修改正式数据
python dataset/sis_dataset_generation/validate_generation.py
```

修改config.yaml中的modes选择train、perturbation、classification。train和classification共用随机ER/SF母网络与增广函数；扰动两条路径共享初始网络，各快照独立仿真。
classification_splits控制原监督划分数量（偶数）；scaling_sizes/scaling_topologies/scaling_count控制规模组。只生成监督数据时设scaling_sizes: []；只生成规模数据时将三个split数量设为0。

默认输出根为dataset；--output可指定其他数据根。先在该根的.staging/<run_id>生成并校验，成功才归档旧目标到archive/<run_id>后发布；失败保留暂存记录，不替换正式数据。原SIS生成结果仍留在原位置，本次未迁移数据。

输出：SIS、SIS_test、SIS_supervised/<split>、SIS_scaling/<ER或SF>/N<n>。
文件名不变：As为float32[N,N]；N<large_graph_threshold（默认2000）保存稠密数组，更大网络保存SciPy CSR矩阵。numes仍为float64[16,20,N]；监督必存rs、basers、x_last，扰动使用xlast_<类型>_<策略>。小规模保持原加载器兼容；大规模须使用已适配的benchmark_sis_scaling.py。
GBB值必须保存；gbb_times可选，秒数，仅供生成阶段记录，规模评估重新计时。

默认SIS方程、16初值、linspace(0,20,200)前20点不变。method:auto在N<2000时使用LSODA，更大网络使用显式RK45，避免隐式求解器的稠密Jacobian；后者为满足终态残差会收紧积分容差。两者均执行原变化量和残差检查，不收敛不打标签。真实标签由平均终态>=0.001判定，GBB由infection_rate*beta_eff>delta判定。无边网络GBB=0。监督候选N<=20不接收，与原加载器一致；类别分别配额接收，达到尝试上限报错。

### 大规模生成（最高30000节点）

设置scaling_sizes: [30000]可单独生成最大规模；完整规模列表可自行增加。建议一次请求一个规模，scaling_count仍为每种拓扑的总样本数（必须为偶数），classification_delta仍是闭区间随机整数范围，不根据目标类别改变候选参数。达到配额无法保证：固定delta或不合适的范围可能只产生一类。

- 大规模仍使用相同ER/SF母网络分布及增广方式，但仅生成可能匹配目标N的删节点数量（保留原奇数删除数量规则），删边搜索每母网络最多max_edge_augmentations次（默认128）。这不是历史随机数序列的逐值复现，也不是枚举全部增广；截断次数记录在generation.json。GCC小于目标N的母网络直接跳过。
- 单个删节点比例100次均无边时跳过该比例；超过max_sf_stubs的SF母网络计入母网络尝试数并拒绝。求解失败、超时、不收敛则停止，绝不跳过这些样本以凑类别配额。
- CPU串行求解。trajectory_timeout_seconds限制每条轨迹墙钟时间（默认1800秒），max_T限制动力学积分时间；两者含义不同。实际耗时依赖拓扑和参数，30k节点不等于所有任务都能在固定时间完成。
- 接收样本先逐个存于.staging/<run_id>/.samples，随后流式写成原顶层list pickle；generation.json中的pickle_index支持逐样本读取。迁移数据时应复制整个组，包括JSON。不要用pickle.load一次加载整个大规模轨迹列表；原小规模加载器仍可正常读取。
- 成功发布后清理本次暂存样本副本；失败时保留供诊断（不自动续算）。暂存期间需同时容纳样本副本和最终文件，并留有至少1GiB磁盘余量。单个30k样本的16×20轨迹约73.2MiB，不能只按CSR大小估计磁盘空间。
- 设置单次内存分配保护，超出安全余量会明确报错，而非继续申请。规模评估也按样本读CSR；ResInf本身仍需要稠密拓扑，数据可生成并不代表所有模型都能在30k节点推理。

实际30k测试（临时目录，不写正式数据）：
```bash
python dataset/sis_dataset_generation/validate_generation.py --large-smoke-test
python dataset/sis_dataset_generation/validate_generation.py --scaling-cache-test
```

每组generation.json保存版本、完整配置、源码指纹、来源及样本统计，sis_params.json保存逐样本参数。隔离保证限定在本次运行的母网络种子命名空间及有序邻接去重，不声称排除与历史数据的重叠。
规模生成后评估配置位于打印的.staging/<run_id>/scaling_evaluation.yaml，默认10次模型重复。

2026-09-08验证：原小规模加载接口、配额、CSR及标准pickle/索引读取、拓扑生成等价性、求解器对照、重试/超时边界和发布前失败保护通过；Neuronal原6项测试通过。临时真实ER/SF各30000节点、16条轨迹及终态检查通过：单个邻接文件分别0.663/1.286MiB，测试进程采样峰值RSS约636.8MiB；单样本仿真与I/O分别约12.69/368.61秒。这是本机指定样本的测试结果，不是运行时间或内存上界，不作为论文性能数据。未生成或替换正式数据。
