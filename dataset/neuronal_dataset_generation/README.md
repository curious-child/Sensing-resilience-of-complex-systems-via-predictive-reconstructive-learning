# Neuronal统一数据生成

```bash
python dataset/neuronal_dataset_generation/generate_neuronal_dataset.py --config dataset/neuronal_dataset_generation/config.yaml
python dataset/neuronal_dataset_generation/test_generation.py
```

在项目根目录运行；modes选择train、perturbation、classification；默认只配置监督24/12/12样本，不会在导入或测试时自动生成。--output可指定隔离的数据根，默认dataset。暂存/验证/归档发布复用相邻SIS生成器，不依赖迁移前目录。

**一套网络、一次仿真、两种标签。** 每个划分按三状态0/1/2配额1:1:1；binary_rs=int(three_class_rs==2)，所以二分类自然2:1。GBB同样从一个三分类结果派生二分类。不创建两套平衡目录；二分类训练子集由监督实验抽样。

输出为Neuronal、Neuronal_test及Neuronal_supervised/{train_dataset,val_dataset,test_dataset}。
监督六个必需文件：As.pkl、numes.pkl、binary_rs.pkl、three_class_rs.pkl、binary_basers.pkl、three_class_basers.pkl；可另存x_last.pkl。全部顶层list、逐项同步。监督邻接float64、轨迹[16,100,N]；训练/扰动邻接float32、轨迹[16,20,N]。终态float64[16,N]。
训练/扰动(mu,d)=(3,1.5)，监督(3.5,2)，方程为-x+A@sigmoid(d*x-mu)。保留全0、14条log-uniform同值、全5初值及原采样网格；RK45和收敛规则见配置。训练只积分到观测末点，终态检查失败报错不打标签。

操作性真标签：终态轨迹节点均值极差>3为1；否则最小值>3.5为2；其余为0。有限初值不等同证明全部吸引子。
critical_values(mu,d)要求mu>2,d>0；缓存双折叠阈值，边界相对容差1e-10归中间类，无边为0。gbb_critical_values.csv保存ResInf补充表4九组上阈值对照及对应下阈值，计算使用全精度，不硬编码6.61。GBB计时可选且排除已缓存阈值计算。

源码共享SIS拓扑；各用途/划分使用不同母网络种子，记录有序邻接跨组去重，不保证与缺少来源的历史数据无重叠。完整数据组通过校验才归档替换，失败及未请求划分不替换。

微型测试覆盖六个文件对应、1:1:1与派生2:1、原二/三分类加载器、九组阈值/平衡点稳定性、原方程/网格、收敛失败、发布失败回滚、归档及版本检查。未运行正式生成。
