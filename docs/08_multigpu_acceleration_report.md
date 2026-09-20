# 纯 Transformer C (1.5B) 四卡 (GPU 4-7) 加速实测与性能分析报告

> 2026-09-20 通信诊断补充：下文2.45秒已在GPU4–7复现，但不是固定物理下限。
> 同步相同6.061GB FP32梯度，改用GPU0/2/4/6后为1.359秒；全部28个卡对与并发DMA
> 指向共享PCIe上行争用。P2P可用性还依赖卡组和路径，不能仅依据驱动能力查询。
> 参见[八卡通信实测](09_eight_gpu_communication_diagnosis.md)。下文保留原实验记录；
> “物理硬墙”和梯度累积“最优解”的表述应结合新证据理解。

本报告记录了在 Tesla V100S-PCIE-32GB 物理卡 4, 5, 6, 7 上对 Qwen3-1.5B 密集纯 Transformer C 架构进行的一系列基准压测、算子消融与分布式加速实验结果。

---

## 一、测试环境与基准配置

- **计算设备**：4× Tesla V100S-PCIE-32GB（物理 GPU 4, 5, 6, 7）
- **互联拓扑**：PCIe 总线互联（无 NVLink，双向理论带宽约 16 GB/s，实测有效带宽约 3.7 GB/s）
- **模型规格**：Qwen3-1.5B 纯 Transformer C 架构（1,515,243,008 参数，30 层，宽度 2048，SwiGLU 6144，Q/K-Norm，Patch 4×4 编码器，128 步字节解码头）
- **数据规范**：`chinese_document_v3` 上下文 1024 打包文档流，每卡 batch_size=4（全局 batch_size=16）
- **基础加速库**：PyTorch 2.8.0+cu128, xFormers 0.0.32.post2, Fused AdamW, Torch Inductor

---

## 二、基准测试结果 (Step 0: Baseline)

在默认全层梯度检查点（30 层全部重算）与 256 MiB 梯度分桶同步配置下的实测数据：

| 评测指标 | 实测数值 | 说明 |
|---|---|---|
| **全局吞吐 (Global Targets/s)** | **4,142.11 targets/s** | 每秒有效预测位置数 |
| **单步总耗时 (Step Seconds)** | **3.955 秒** | 包含前向、反向、梯度同步与权重更新 |
| **梯度同步耗时 (Gradient Sync)** | **2.451 秒** | 4 卡 AllReduce 传输 6.06 GB 梯度耗时 |
| **通信耗时占比 (Sync Fraction)** | **61.98%** | **PCIe 通信是绝对主瓶颈** |
| **纯计算耗时 (Compute Seconds)** | **1.504 秒** | 前向与反向实际计算时间 |
| **单卡显存峰值 (Peak Allocated)** | **23.64 GiB** | 显存保留值约 26.88 GiB |

---

## 三、四大方向优化消融与对比分析

### 优化 1：分布式通信与分桶调优 (Distributed Sync)

- **实验设计**：对比单缓冲串行分桶、双缓冲流水线 AllReduce (`--double-buffer-sync`) 以及 1024 MiB 大分桶 (`--gradient-bucket-mib 1024`)。
- **实测数据**：
  - **默认 256 MiB 串行分桶**：同步耗时 2.451 秒，吞吐 4,142.11 targets/s
  - **双缓冲重叠流水线**：同步耗时 2.552 秒，吞吐 4,034.40 targets/s（显存略增 250 MB）
  - **1024 MiB 大分桶**：同步耗时 **2.448 秒**，吞吐 4,137.93 targets/s
- **物理机理归因**：
  在 1.5B 参数规模下，单卡 FP32 梯度达 6.06 GB，4 卡 Ring-AllReduce 的总物理通信量为 $2 \times \frac{3}{4} \times 6.06 = 9.09 \text{ GB}$。在无 NVLink 的 PCIe 拓扑中，单向有效带宽受限于 ~3.7 GB/s，通信时长存在 $9.09 / 3.7 \approx 2.45$ 秒的物理硬墙。D2D 内存拷贝（6 ms）仅占千分之二，因此软件层流水线无法绕过物理总线极限。
  **工程结论**：在 PCIe 硬件下，单步增大有效计算量（调大 Batch 或使用 Gradient Accumulation 累积多步合并一次通信）是分摊通信开销的最优解。

---

### 优化 2：自定义算子融合与头损失优化 (Byte Loss Kernel)

- **实验设计**：测试静态常量预分配、Fused CE & Mask Reduction，并对比 Triton Autotuning (`mode="max-autotune"`) 与默认 Inductor 编译。
- **实测数据**：
  - **默认 Inductor 编译 (cuBLAS GEMM)**：**271.30 ms / step**
  - **Max-autotune (Triton Autotune)**：**280.32 ms / step** (吞吐为默认的 0.97x)
- **物理机理归因**：
  Tesla V100S 为 Volta 架构（`sm_70`），片上共享内存为 96 KB。OpenAI Triton 主要针对 Ampere (`sm_80`) / Hopper (`sm_90`) 进行大分块流水线优化，在 Volta 上容易受限于 shared memory 溢出且其生成的 Triton GEMM 性能落后于 NVIDIA 官方汇编手写的 cuBLAS 库。
  **工程结论**：在 Volta 显卡上，保持 PyTorch Inductor 调用 cuBLAS 配合 Elementwise 自动融合是最高效且稳定的算子组合。

---

### 优化 3：选择性激活值检查点 (Selective Checkpointing)

- **实验设计**：在 batch_size=2 显存余量充足条件下，对比全层检查点（30 层全重算）与交替选择性检查点（`step=2`，15 层重算、15 层保留激活值）。
- **实测数据**：
  - **全层重算 (Full Checkpointing)**：
    - 全局吞吐：2,499.34 targets/s
    - 纯计算耗时：**0.8259 秒**
    - 显存峰值：Allocated 23.24 GiB / Reserved 25.70 GiB
  - **交替选择性重算 (Selective Checkpointing step=2)**：
    - 全局吞吐：**2,544.80 targets/s**
    - 纯计算耗时：**0.7681 秒**（**纯计算耗时直接降低 7.0%**）
    - 显存变化：Allocated 28.00 GiB / Reserved 30.12 GiB（安全处于 32GB 显存之内）
- **工程结论**：
  通过跳过 15 层的 SwiGLU MLP 前向重算，显著降低了 GPU 浮点计算量；但由于 PCIe 梯度同步的 2.45s 串行等待占了单步 76%，整体吞吐呈现稳步提升。

---

### 优化 4：内层 128 步解码优化与加速 (Inference Rollout)

- **实验设计**：针对生成推理中 128 步内层字节解码循环（原先占用 rollout 总时长的 91.8%），实现预分配静态 KV Cache 与零校验调度器（`generate_fast`），并在真实 Qwen3-1.5B 权重（`positions_1089139385.pt`）上进行 16 格真实文本续写生成压测。
- **实测数据**：
  - **标准 Rollout (Standard)**：
    - 生成吞吐：**2.395 glyphs/s**
    - 内层字节解码耗时：**3.068 秒**（占 rollout 总耗时的 **91.82%**）
  - **静态高速 Rollout (Fast Decode)**：
    - 生成吞吐：**2.681 glyphs/s**（**端到端整体提速 +11.9%**）
    - 内层字节解码耗时：**2.707 秒**（每 8 字解码耗时缩短 361 ms）
    - **一致性校验**：与标准自回归输出在 8 个验证提示上保持 **100% 位级严格对齐，0 像素误差**。
  - **单算子解码微基准**：
    - 独立内层字节循环（脱离 1.5B 骨干前向）：从 3.20 glyphs/s 跃升至 **24.23 glyphs/s**（**+7.57 倍单算子加速**）。

---

## 四、加速总结与实战建议

| 优化方向 | 改进手段 | 测量维度 | 提升幅度 |
|---|---|---|---|
| **方向 1：通信调优** | 1024 MiB 分桶调优 | 4 卡同步时间 | 2.451s → 2.448s (PCIe物理极限) |
| **方向 2：算子融合** | Inductor + cuBLAS 融合 | 字节头损失计算 | 优于 Triton 动态调优约 3.3% |
| **方向 3：计算重算** | 15 层交替选择性检查点 | 纯计算反向传播耗时 | **纯计算提速 +7.0%** (0.826s → 0.768s) |
| **方向 4：推理生成** | 静态 KV Cache 与快速循环 | 单字 Rollout 端到端吞吐 | **吞吐提速 +11.9%** (单算子提升 7.57 倍) |

> 🚀 **多卡训练终极提速建议**：
> 在当前 4 卡 PCIe 架构下，由于单步通信时间恒定为 2.45 秒，最强大的训练加速杠杆为**使用 `gradient_accumulation_steps = 2`**（累积 2 步合并一次 PCIe 通信）：
> - 单步有效目标翻倍至 32,768 targets；
> - 预计整体训练吞吐可直接从 **4,142 targets/s 跃升至 ~6,000 targets/s**（**端到端提升 +44.9%**），最大化压榨 V100 计算潜能。
