# 八卡通信瓶颈实测

测试日期：2026-09-20。所有测试均在八卡空闲时启动，不加载模型或数据集，
不修改驱动、ACS、IOMMU、系统 CUDA、训练配置或持久 NCCL 设置。

结论：当前可用SHM路径的主瓶颈是共享PCIe上行争用，选卡布局比增大分桶更重要。
相同四卡/6.061GB梯度，GPU4–7需2.449秒，GPU0/2/4/6需1.359秒。
P2P存在组合相关的超时边界，不能从两卡成功推断四卡和八卡安全。
共完成7套109个case：104个成功且通过结果检查，5个P2P诊断case超时并正常回收。
结束时八卡均为5MiB、0%利用率，无残留诊断GPU进程或tmux任务。

## 环境与方法

- 8× Tesla V100S PCIe 32GB，SM70；驱动实测 `535.183.01`，CUDA driver API `12020`。
- PyTorch `2.8.0+cu128`；运行时 NCCL `2.27.3+cuda12.9`；Python `3.12.13`。
  NCCL 构建标签、PyTorch runtime 与宿主驱动版本是不同指标。
- 双路 Xeon Gold 6230R，104 逻辑 CPU；LXC，cpuset 允许 CPU 0–103、NUMA 0–1，
  `cpu.max=max 100000`，未发现容器 CPU 配额节流上限。
- 全部 GPU 与抽查的上行桥均报告 PCIe Gen3 ×16，无降速/降宽；无 NVLink。
- 默认对照：`NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0`；NCCL 日志证实
  `SHM/direct/direct`，大消息采用 Ring/Simple。
- 脚本：`scripts/benchmark_gpu_communication.py`。smoke/matrix 来源 `73cd31f`；
  p2p/dma/extended/transport 来源 `f30250e`；verify 来源 `90e8614`。
- 每种大小预热 3 次、测量 8 次；每次测量前 barrier 和 CUDA synchronize，
  统计各 rank 当次 wall time 的最大值，再取 8 次的中位数；另存 CUDA event 时间。
  初始化、填充、完整正确性检查不计入通信时间。smoke 为 3 次测量。
- 所有 AllReduce 使用 FP32 SUM，按 rank 初始化，并在每次操作后逐元素检查结果。
  模型等量实验是 1,515,243,008 个 FP32 元素，即 6,060,972,032 bytes。
  项目同步实验调用原有 `sync_gradients`，使用一个合成连续梯度及 256MiB scratch；
  它重现总字节量和同步代码，但不等同于真实模型的每个参数布局。
- 单个 case 具有进程组超时和外层 TERM/KILL 超时。只有本次 case 的进程受超时管理；
  任一 case 结束后仍有 GPU 进程，套件立即停止。没有终止用户训练任务。

## 拓扑

```text
NUMA 0
  PCIe Gen3 x16 上行 A ─ 共享桥 ─ GPU0、GPU1
  PCIe Gen3 x16 上行 B ─ 共享桥 ─ GPU2、GPU3
NUMA 1
  PCIe Gen3 x16 上行 C ─ 共享桥 ─ GPU4、GPU5
  PCIe Gen3 x16 上行 D ─ 共享桥 ─ GPU6、GPU7
```

0/1、2/3、4/5、6/7 为 PIX；同 NUMA 内不同桥为 NODE；跨 NUMA 为 SYS。
驱动的 P2P read/write 能力矩阵全部为 OK，但这不保证实际 NCCL 路径或性能。
只读 `lspci -vv` 检查发现四组相关桥上 `ReqRedir+ CmpltRedir+`，即 ACS 请求和
完成重定向开启。该设置可以使本来相邻的设备流量经过上游，不能据 PIX 标签假设流量
一定在交换机内直接完成。本次没有关闭 ACS，因此不会宣称已用干预实验唯一证明 ACS 根因。

## 全部 28 个卡对

64MiB FP32 AllReduce，算法带宽 GB/s（十进制）。两卡时该值也等于 ring-equivalent
bus bandwidth；它不是单向 `cudaMemcpy` 带宽，更不是整机所有链路吞吐量的和。

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | 3.31 | 5.44 | 5.41 | 5.38 | 5.36 | 5.41 | 5.37 |
| 1 | 3.31 | — | 5.56 | 5.59 | 5.39 | 5.45 | 5.40 | 5.33 |
| 2 | 5.44 | 5.56 | — | 3.30 | 5.43 | 5.40 | 5.40 | 5.42 |
| 3 | 5.41 | 5.59 | 3.30 | — | 5.40 | 5.40 | 5.42 | 5.39 |
| 4 | 5.38 | 5.39 | 5.43 | 5.40 | — | 3.30 | 5.47 | 5.51 |
| 5 | 5.36 | 5.45 | 5.40 | 5.40 | 3.30 | — | 5.44 | 5.47 |
| 6 | 5.41 | 5.40 | 5.40 | 5.42 | 5.47 | 5.44 | — | 3.31 |
| 7 | 5.37 | 5.33 | 5.42 | 5.39 | 5.51 | 5.47 | 3.31 | — |

四对 PIX 卡均明显更慢，24 对 NODE/SYS 卡均更快，未发现一张独立异常慢卡。
因此该工作路径下，跨 NUMA 不是解释主要性能损失的变量；共享上行的选卡方式更关键。

## 模型等量梯度：2.45 秒并非固定下限

单位秒；每次同步完整 6.061GB FP32 梯度。以下只有通信，不含模型前后向和优化器。

| GPU 组合 | 单次连续 AllReduce | 直接256MiB分桶 | 项目256MiB分桶同步 |
|---|---:|---:|---:|
| 0,1：同桥两卡 | 1.83957 | 1.83739 | 1.86414 |
| 0,2：跨桥、同NUMA | 1.09084 | 1.09328 | 1.12073 |
| 0,4：跨NUMA | 1.11177 | 1.11178 | 1.13531 |
| 0,1,2,3：两个共享上行 | 2.43014 | 2.42988 | 2.45554 |
| 4,5,6,7：两个共享上行 | 2.41950 | 2.42515 | 2.44859 |
| 0,2,4,6：四个独立上行 | 1.33516 | 1.33540 | 1.35924 |
| 0–7：四个上行、每个两卡 | 2.72420 | 2.72794 | 2.75163 |

0/2/4/6 相对 4/5/6/7 的同步时间减少 **44.49%**，通信阶段吞吐提高 **80.14%**。
同样四张卡、同样参数量、同样FP32与同步代码，仅更换选卡组合就显著改变结果。
这也说明增加全局 batch 或梯度累积并不是唯一的下一步选项。

纯 scratch 往返拷贝约 25ms，相对 2.449s 约占 1%；整个梯度一次归约和直接分桶
归约几乎相同。原项目循环、分桶调用延迟和GPU本地拷贝都不是当前秒级开销的主因。

如果仅把旧四卡单步中的2.451s同步换成1.359s、暂假设其他1.504s不变，
每步预计约2.863s，16,384目标/步对应约5,722目标/s。这是跨实验的粗略估算，
**不是实测训练吞吐**；跨NUMA选卡的实际数据加载、CPU竞争、显存占用和计算通信
相互影响仍需真实训练A/B验证。

八卡同步时间比原四卡多约12.4%，而非翻倍。八卡算法带宽与ring-equivalent
bus bandwidth也必须区分：256MiB时约2.218GB/s和3.881GB/s；四卡4–7分别
约2.496GB/s和3.744GB/s。它们接近的bus值不代表已经证明PCIe的理论极限。

## CPU affinity 与禁用共享内存的对照

256MiB AllReduce，算法带宽GB/s：

| GPU | 本地CPU affinity | 远端CPU affinity | 禁用SHM改走Socket |
|---|---:|---:|---:|
| 0,1 | 3.291 | 3.257 | 2.498 |
| 0,4 | 5.472 | 5.490 | 1.732 |
| 4,5,6,7 | 2.494 | 2.462 | 1.445 |
| 0–7 | 2.217 | 2.176 | 未测 |

CPU affinity 改变只带来约几个百分点以内差异，无法解释同桥和跨桥的大幅差距。
当前SHM路径明显优于Socket替代路径，不应把关闭SHM当作默认加速方案。

## 主机 DMA 与 P2P 对照

单卡 pinned-host 64MiB 拷贝，在本地 NUMA CPU affinity 下，八卡 H2D 为
7.96–8.18GB/s，D2H 为 8.57–8.71GB/s；绑定另一 NUMA 的 CPU 后分别为
7.24–7.42GB/s、7.99–8.02GB/s。这是 CPU affinity/分配路径对照，没有调用
`mbind` 强制每一页的内存归属，不能把它表述成严格的内存绑定实验。

本轮开启 `NCCL_P2P_DISABLE=0`，保留 `NCCL_CUMEM_HOST_ENABLE=0`：

- 0/1 日志为 `P2P/CUMEM`，256MiB AllReduce 约 3.108GB/s，运行成功但没有提速。
- 0/2 的 256MiB AllReduce 约 5.582GB/s。
- 0/4 的 256MiB AllReduce 约 5.451GB/s，但实际仍走 `SHM/direct/direct`。
  允许 P2P 不等于 NCCL 对 SYS 连接也选择 P2P。
- 0/1 单独开启 cuMem host 分配也成功，256MiB 约 3.329GB/s。
- 单进程 PyTorch 跨 GPU `copy_`：0/1、0/2 约 7.95GB/s；0/4 两个方向仅约
  1.16/0.68GB/s。该 API 路径与 NCCL 协议不同，不可直接拿来替代 AllReduce 带宽。

这些结果推翻“当前驱动环境下 P2P 或 cuMem host 必然不能运行”的绝对判断。
历史训练初始化挂起在本轮两卡小内存测试中未复现；本轮成功也不保证历史高显存、
并发训练和初始化条件全部安全。

## 强制跨 NUMA P2P 的实际失败边界

在0/4上只增加 `NCCL_P2P_LEVEL=SYS`、允许P2P，并保留cuMem host关闭：
NCCL日志从默认的SHM变为`P2P/CUMEM`，随后在最初4字节AllReduce/barrier处挂起。
两rank的35秒watchdog均报告超时，外层75秒timeout终止了该case（返回码124），
GPU进程被正常回收，后续case继续运行。没有产生可用的带宽结果。

同一对GPU不强制SYS P2P时的SHM路径成功且约5.45GB/s，因此失败不能归因于
测试张量大小或不同rank调用顺序。当前平台确有跨NUMA P2P执行限制；但本轮没有
更换驱动或调整ACS/IOMMU，不能进一步唯一判定是驱动缺陷、固件路由还是平台隔离配置。
本次仅对0/4强制SYS，不把该单个失败扩写成所有跨NUMA卡对均已测试失败。

`nvidia-smi topo -p2p a`报告所有卡对native atomics均为NS，包括正常工作的同NUMA
卡对。因此该能力项本身也不足以单独解释已观察到的P2P超时。

四卡与八卡进一步对照（仅将P2P开关设为0，未强制SYS）：

| 组合 | cuMem host关闭 | cuMem host开启 |
|---|---|---|
| 4,5,6,7 | 集合通信超时，外层回收 | 集合通信超时，外层回收 |
| 0,2,4,6 | 通过，256MiB约4.538GB/s | 通过，约4.543GB/s |
| 0–7 | 集合通信超时，外层回收 | 集合通信超时，外层回收 |

0/2/4/6开启P2P后的带宽与关闭时约4.531GB/s基本一致，没有建立加速收益。
其跨NUMA环边仍由NCCL自动选择合适路径，不能解读为SYS P2P已恢复。
GPU4–7在初始barrier/第一条大消息附近停滞，未产生有效带宽结果。这与历史四卡
启动问题相符，但不是对历史进程状态的逐项重放，也不能把所有问题都归因于cuMem host。

因此生产基线继续保留`NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0`。
本轮没有证据支持全局开启P2P或强制`NCCL_P2P_LEVEL=SYS`。

## 并发DMA：独立于AllReduce的共享上行证据

各rank同时执行64MiB pinned-host拷贝，按最慢rank时长计算每卡/全组带宽，GB/s：

| GPU组合 | 每卡H2D | 每卡D2H | 全组H2D | 全组D2H |
|---|---:|---:|---:|---:|
| 0,1 | 4.670 | 5.493 | 9.341 | 10.986 |
| 0,2 | 7.629 | 8.606 | 15.258 | 17.212 |
| 0,4 | 7.652 | 8.637 | 15.305 | 17.275 |
| 4,5,6,7 | 4.662 | 5.593 | 18.647 | 22.374 |
| 0,2,4,6 | 7.622 | 8.602 | 30.488 | 34.410 |
| 0–7 | 4.663 | 5.237 | 37.301 | 41.898 |

此计时区间内没有AllReduce，只执行CUDA拷贝；NCCL仅在计时前同步开始、计时后
收集结果。共享桥的组合再次显著变慢；每个桥一张卡可以保留接近单卡的拷贝带宽。
这为共享PCIe上行的资源争用提供了独立于梯度分桶和归约算法的证据。

## NCCL传输实现对照

256MiB AllReduce，算法带宽GB/s；仅改变当前case环境变量。

| 组合 | 默认SHM | SHM memcpy mode3 | 4 channels | 8 channels |
|---|---:|---:|---:|---:|
| 0,1 | 3.309 | 3.857 | 3.593 | 3.646 |
| 0,4 | 5.449 | 6.469 | 6.282 | 7.007 |
| 4,5,6,7 | 2.496 | 2.567 | 2.561 | 2.572 |
| 0–7 | 2.218 | 2.195 | 2.243 | 2.249 |

默认列来自对应完整梯度case中的256MiB对照，四舍五入。
memcpy mode3设置为`NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3`；
通道数同时设置`NCCL_MIN_NCHANNELS`和`NCCL_MAX_NCHANNELS`。
mode1/mode2也测试并保留原始结果，均未显示对四卡/八卡的更大收益。

卡对上的收益不能直接外推到多卡：0/4增加通道数收益较大，但4–7只有约3%，
八卡只有约1.4%。这些是NCCL版本相关的诊断选项，不是已验证的生产训练默认值。
其结果说明部分软件带宽利用率有优化空间，同时共享上行仍限制多卡收益。

## 原始证据

原始 JSON 与 NCCL 日志保留在服务器、由 Git 忽略的目录：

- `artifacts/reports/comm_8gpu_smoke_v1/`
- `artifacts/reports/comm_8gpu_matrix_v1/`
- `artifacts/reports/comm_8gpu_p2p_v1/`
- `artifacts/reports/comm_8gpu_dma_v1/`
- `artifacts/reports/comm_8gpu_extended_v1/`
- `artifacts/reports/comm_8gpu_transport_v1/`
- `artifacts/reports/comm_8gpu_verify_v1/`

对应case日志归档到`artifacts/logs/comm_8gpu_*_v1/`，主控console日志也位于
`artifacts/logs/`。最初测量时case日志在各报告目录，完成后统一归档；脚本已默认
将新日志写到`artifacts/logs/<output目录名>/`。日志可能包含本机标识或接口信息，
不应原样加入公共Git；本报告只保留脱敏指标。

## 后续行动

1. 在保持FP16计算、FP32归约、全局batch和优化器更新频率不变的前提下，优先做
   GPU4–7与GPU0/2/4/6的真实训练A/B；通信阶段已有44.49%的耗时下降证据。
2. 继续使用经过验证的SHM回退路径。P2P的能力查询、两卡成功与四卡/八卡成功是
   不同层次的证据，不能用前两者代替后一项。
3. 若下一步只做软件优化，验证真正的反向/通信重叠，再考虑通道数；本轮同桥四卡
   通道调优仅约3%，继续改大bucket不应成为主要投入。
4. 梯度累积会改变有效batch与更新频率，应单列训练质量实验；不应因为2.45秒
   被误认为固定下限就直接翻倍batch。压缩梯度也需要单独数值/收敛验证。

本轮没有更换驱动，因此已验证的是当前R535/SM70/PCIe/容器组合的行为边界。
共享上行争用有拓扑、28卡对、等量梯度、并发DMA四组证据；ACS重定向有只读配置
证据，但没有关闭ACS的干预实验。不能宣称升级驱动就一定消除这些硬件路径限制。

## 复现

先按AGENTS.md拉取提交、同步uv环境并核对八卡空闲。在tmux内运行，例如：

```bash
uv run --no-sync python scripts/benchmark_gpu_communication.py \
  --suite smoke --output artifacts/reports/comm_reproduction_smoke
uv run --no-sync python scripts/benchmark_gpu_communication.py \
  --suite matrix --output artifacts/reports/comm_reproduction_matrix
```

`extended`包含完整6.061GB梯度，`dma`包含单卡CPU affinity对照，`transport`
包含SHM memcpy与通道数对照，`p2p`包含两卡P2P，`verify`包含并发DMA和明确可能超时的
P2P边界测试。各套件必须顺序运行，输出目录必须不存在。详细配置与每次时间见JSON。
