---
title: "RTX 4090 Tensor Core + PCIe P2P 双解锁：两台四卡服务器的部署与实测"
published: 2026-10-27
description: "从 Warewulf 无状态启动、Ada 固件前置解锁到 NVIDIA Open P2P、32GiB BAR1 和 NCCL 实测：一次可以复盘的 RTX 4090 GPU 节点改造。"
tags: [RTX 4090, NVIDIA, CUDA, P2P, NCCL, Warewulf, HPC, Linux]
category: "HPC / GPU"
lang: "zh-CN"
draft: false
pinned: true
image: "/images/posts/rtx4090-unlock/cover.svg"
comment: true
---

> **说明与脱敏**：本文基于真实完成的双节点实验，主机名统一替换为 `compute-a` / `compute-b`，不包含真实域名、IP、MAC、个人账户、管理接口与设备 PCI BDF。所有管理和部署路径均为示例，请按自身环境替换。固件解锁与 NVIDIA 驱动补丁均非官方支持功能，可能引发设备不可用、数据错误、安全隔离问题或硬件风险；不要在缺乏带外恢复与回滚能力的生产环境直接执行。

## 先看结果

我们在**两台各有 4 张 RTX 4090 24GB** 的 GPU 服务器上，结合 [gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) 与 [Duanyll 的 P2P 补丁](https://github.com/Duanyll/open-gpu-kernel-modules) 完成了两类改造：

- Tensor Core 初始化：每台的 4 张卡均出现 `gpu_unlock: unlocked`；在特定 MMA 模式中，FP16/FP32 从 **161.8** 增至 **323.6 TF/s**，BF16/FP32 从 **161.7** 增至 **323.4 TF/s**，接近 **2 倍**。
- BAR1：每卡由 **256 MiB** 扩至 **32768 MiB（32 GiB）**；这不是增加显存容量。
- CUDA P2P：两台服务器共 **24 条有向卡对**全部数据校验通过，`errors=0`，单方向 64MiB 拷贝吞吐约 **16–21 GiB/s**。
- NCCL：单机四卡 AllReduce 全部校验 `#wrong=0`。32–256MiB 下，**默认 AUTO 拓扑策略约 18.5 GB/s**，快于手动指定 PHB 与 SYS。
- 测试边界：这不是跨服务器八卡 NCCL 测试，也没有完成长期高负载稳定性认证。

**最容易误解的地方**：特定 MMA 指令 2 倍，不等于 GEMM 或 AI 训练整体 2 倍；`topo -p2p` 显示 OK，也不等于 NCCL 一定选 P2P，更不代表全局应用必然加速。


![双节点 GPU、Tensor 解锁与 PCIe P2P 组合架构](/images/posts/rtx4090-unlock/cover.svg)

与“复制命令然后宣布成功”不同，这次实验围绕三个实际问题展开：**消费级 Ada 显卡究竟限制了什么、PCIe P2P 卡在哪里、为什么有些 NCCL 测试打开 P2P 后反而更慢？** 我们逐一以固件初始化日志、PCIe 资源分配、CUDA 数据校验与 NCCL 重复实验回答。

:::note[如何阅读]
本文的配置和命令是经过脱敏的实验复盘，不是任何主板或驱动版本的通用“一键脚本”。先看机制与风险，再结合自己的 PCIe 拓扑、GSP/驱动版本和启动链修改。
:::

## 1. 实验环境与双解锁机制

| 项目 | 配置 |
| --- | --- |
| 节点 | 2 台，每台 4 × RTX 4090 24GB |
| CPU 拓扑 | 双路 Xeon；GPU0/1、GPU2/3 位于不同 NUMA 域 |
| 计算节点 | Fedora 44，内核 `6.19.10-300.fc44.x86_64` |
| 管理/启动 | Warewulf 4.7，dracut + wwinit，Slurm |
| 旧驱动 | NVIDIA Open 610.57.04 |
| 新驱动 | NVIDIA Open 615.71.09 + P2P Patch |
| 固件解锁 | gpu-unlock 1.1.1 |
| CUDA Toolkit | 13.3；驱动升级后 CUDA UMD 显示 13.4 |
| NCCL | 2.31.2；nccl-tests 2.21.1 |

两个计算节点没有互联网。所有源码下载、RPM 准备、镜像处理均在联网管理端完成，计算节点只运行本地测试，再把日志拉回持久存储。

### 1.1 Tensor Core 是初始化阶段的限制

[gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada) 并不是传统意义上的超频。它通过预先生成的 AD102 固件资产和 `gpu_unlock.ko`，在 NVIDIA 驱动接管 GPU **之前**完成特定速率配置。正确的逻辑顺序是：

```text
冷启动 → initramfs 内 gpu_unlock + AD102 资产
      → NVIDIA Open Kernel Modules → CUDA / cuBLAS / NCCL
```

普通系统运行后再加载模块，不等价于在 GPU 固件初始化前介入。固件资产的取得和再分发必须遵守上游项目与原始许可证。

### 1.2 P2P 是 GPU 显存映射与驱动能力问题

RTX 4090 没有 NVLink。原始驱动即使枚举了四张卡，`nvidia-smi topo -p2p r/w` 仍可能全部显示 `GNS`。P2P 驱动补丁修改 NVIDIA Open 模块的 GPU peer mapping 支持；参考 [作者关于 4090 48GB、动态 BAR1 的技术文章](https://duanyll.com/2026/7/13/4090-48G-P2P/)。

**BAR1 是 PCIe MMIO 可映射窗口，不是额外显存**。GPU 自身报告 ReBAR 最高支持 32GB，并不代表 BIOS 当时实际给它分配了 32GB。本次初始 `current size` 只有 256MB。

### 1.3 从 CPU 到 GPU：为什么 NUMA 位置决定通信上限？

消费级 GPU 之间的 P2P 并不是虚拟的“GPU 私有高速总线”。它仍需走 GPU PCIe Endpoint → Root Port/PCIe Bridge → 对端 GPU 的物理路径。如果两卡落在不同 CPU Socket 的 Root Complex 后面，数据还会涉及处理器之间的互连；启用 P2P 能减少主机内存中转，但不能消除硬件跳数。

![双路 NUMA 下四张 GPU 的拓扑示意](/images/posts/rtx4090-unlock/topology.svg)

在我们的机器上，GPU0/1 和 GPU2/3 是两个相对接近的组合；GPU0→GPU2 属于跨 NUMA 通路。这一硬件结构解释了后面为什么同 NUMA 单向复制达到约 21 GiB/s，而部分跨 NUMA 方向约为 16–17 GiB/s。这里的图是**拓扑示意而非真实机器的 PCI BDF/布线图**。

### 1.4 为什么必须由两个独立机制组合？

Tensor Unlock 发生在 NVIDIA 驱动加载**之前**，目的是更改特定运算速率配置；P2P 补丁则是 NVIDIA Open Kernel Modules 层的 peer mapping 能力。二者在不同层面工作，既不能互相代替，也不意味着理论加速比可以相乘。

这也是我们先用一张卡验证 Tensor，再在克隆镜像中升级 NVIDIA 615 和 P2P、最后检查 GPU 间 NCCL 的原因。按阶段验收能把“启动失败”“GPU 数值错误”“通信错误”定位到不同层。

## 2. 通过 Warewulf 部署 Tensor Unlock

最重要的工程决策是：**永远复制专用镜像，不在整个集群共用的 GPU 镜像上试错**。下列镜像和目录名称都是示例：

```bash
wwctl image copy fedora44-gpu-base fedora44-gpu-tensor
mkdir -p /srv/rtx4090-lab/src
git clone https://github.com/JustAResearcher/gpu-unlock-ada.git /srv/rtx4090-lab/src/gpu-unlock-ada

cd /srv/rtx4090-lab/src/gpu-unlock-ada
git rev-parse HEAD
make inputs
make firmware
make test-full
make rpm
```

本次 `test-full` 执行 33 项测试，30 通过、3 项可选离线解码测试跳过。生成 `gpu-unlock-firmware`、`gpu-unlock-dkms` 两个 RPM，在 Fedora 44 独立镜像里安装。注意 RPM 是在另一发行版构建机上生成的 `noarch` 包，因此不能只看安装成功，要实查 **Fedora 内核版本对应的 DKMS 状态**：

```bash
KVER=6.19.10-300.fc44.x86_64
dkms status
modinfo -k "$KVER" gpu_unlock
cat /usr/lib/dracut/dracut.conf.d/gpu-unlock.conf
```

本次成功后可以看到：

```text
gpu-unlock/1.1.1, 6.19.10-300.fc44.x86_64, x86_64: installed
force_drivers+=" gpu_unlock "
```

**容易踩的坑：Warewulf 的 PXE/wwinit 启动链不等于普通磁盘系统的 GRUB。** 仅把模块放进镜像文件系统并不能确保它在 NVIDIA 驱动之前运行。目标 Fedora 镜像中生成 dracut initramfs，并明确加入 wwinit：

```bash
KVER=6.19.10-300.fc44.x86_64
INITRD=/boot/initramfs-$KVER.img
dracut --force --no-hostonly --add wwinit "$INITRD" "$KVER"
lsinitrd -m "$INITRD" | grep wwinit
lsinitrd "$INITRD" | grep -E 'gpu_unlock|v67-ad102|gsp_boot_ad102'
```

随后退出镜像，使用 `wwctl image build fedora44-gpu-tensor`。实际节点需确认其 iPXE 模式就是 dracut 路径，启动参数确实传入解锁模块。内核模块和 AD102 固件应出现在**真正的早期启动环境**；NVIDIA 完整驱动可以后续加载，不能仅凭早期 initramfs 里没有 `nvidia.ko` 就判为失败。

第一次只针对一张 RTX 4090，在节点专用启动参数中配置 `gpu_unlock.targets=<GPU_BDF>`。BDF 必须从目标机器的 `lspci -Dnn` 查询，**不可把另一台的 PCI 地址照搬过去**。单卡日志确认 `gpu_unlock: unlocked` 后，再扩展为四张。Warewulf 的 `Kernel.Args` 是逗号列表：多 BDF 字符串要作为**一个参数元素**传入，并核对 `/proc/cmdline`，避免默认 profile 参数重复。

初次测试在设备重新上电后完成；内核出现过模块签名未被信任的 taint 日志，但并未阻止解锁。正式 Secure Boot 环境应另行处理模块签名信任链。

## 3. 升级 NVIDIA 615.71.09 并编译 P2P 补丁

### 3.1 驱动版本必须匹配

目标分支为 `615.71.09-p2p-48g`。原环境的 610.57.04 用户态驱动不能直接混装 615 的内核模块。第二个 Warewulf 镜像从**已通过 Tensor 验证的镜像**复制，保留可回滚的旧镜像。

Fedora 44 的 NVIDIA 仓库提供 615.71.09 的用户态组件与 Open DKMS 包。先用精确版本做 `dnf install --assumeno` 事务预演，再安装同版的 `nvidia-driver-common`、`nvidia-driver-cuda`、`nvidia-driver-cuda-libs`、`nvidia-kmod-common`、`kmod-nvidia-open-dkms`、`nvidia-modprobe`、`nvidia-persistenced`。

这里曾踩过一个关键依赖坑：预演时为了避免官方 DKMS 覆盖补丁，排除了 `kmod-nvidia-open-dkms`，却使 `nvidia-kmod-common` 所需的 `nvidia-kmod = 615.71.09` 依赖缺失。解决方法是**先完整升级官方 615.71.09 包组，再单独构建补丁模块**。

另一个细节：RPM 安装成功后 `dkms status` 显示 `nvidia/615.71.09: added`，但 `modinfo` 查询不到 NVIDIA 模块。`added` 只是已注册源码，并不是已构建并安装。

### 3.2 在 Fedora 44 内核开发环境构建

`home` 类型的管理端即使有网，也不能使用管理端 Rocky 内核头文件替 Fedora 44 编译模块。源代码先从管理端只读 bind 到镜像，再复制到镜像本地构建：

```bash
git clone https://github.com/Duanyll/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules
git switch 615.71.09-p2p-48g
git rev-parse HEAD
cat version.mk

# 以下 make 在 Fedora 44 构建镜像内进行
KVER=6.19.10-300.fc44.x86_64
make -j16 modules SYSSRC="/usr/src/kernels/$KVER" SYSOUT="/usr/src/kernels/$KVER"
```

生成并核实版本的模块有：

```text
nvidia.ko
nvidia-modeset.ko
nvidia-uvm.ko
nvidia-drm.ko
nvidia-peermem.ko
```

五个模块的 `modinfo version` 都是 `615.71.09`，`vermagic` 均匹配 `6.19.10-300.fc44.x86_64`。编译中存在 `objtool` 的 `naked return` 警告及缺少 `vmlinux` 导致的 BTF 生成跳过，但链接成功；是否运行正确仍以后续 GPU 实测为准。

本次首次验证把这些补丁模块安装到独立镜像的 `/lib/modules/<KVER>/updates/dkms/`，执行 `depmod -a "$KVER"`，再用 `modinfo -k "$KVER" -F filename nvidia` 验证最终模块优先路径。重新生成包含 `wwinit` 与 `gpu_unlock` 的 dracut initramfs，随后 `wwctl image build`，最后只切换两台已排空的节点。

**不能忽视的运维技术债**：P2P 模块最初是手工放入 `updates/dkms`，而官方 `nvidia/615.71.09` DKMS 源码注册项可能仍存在。将来执行 `dkms autoinstall` 或升级驱动会有覆盖补丁的风险。真正长期维护应封装成版本固定、可重建的补丁 DKMS/RPM，而不是永久依赖一次性手工覆盖。

## 4. BIOS / BAR1 / IOMMU 配置

![BAR1：映射窗口与物理显存的区别](/images/posts/rtx4090-unlock/bar1.svg)

### 4.1 BAR1 扩容不是显存扩容

CPU 可通过 PCIe MMIO 窗口访问 GPU 的部分设备/显存地址空间。BAR1 资源的“size”表示主机为这个 PCIe BAR 分配的**地址窗口容量**，而不是 GPU 搭载了多少显存。大 BAR1 对静态映射覆盖整卡显存非常关键，但它也占用平台的 PCIe MMIO 地址空间；四卡同时选择 32GiB BAR1，理论上需要约 128GiB 的窗口地址资源（还未计入其他 BAR）。

动态 BAR1 映射并非简单“开了 ReBAR 就变成动态”：它还依赖相应的驱动映射策略。我们这里的卡是**标准 24GB**，而原始研究重点包括 **48GB 改装卡**。24GB 容量小于已分配的 32GiB 窗口，因此不能仅凭本次结果证明 48GB 场景下的动态窗口回收算法已被充分验证。本文只声称本次 24GB 环境的 peer copy / NCCL 结果。

### 4.2 从能力位到实际映射有三层检查

1. §lspci -vv§ 的 Resizable BAR **supported sizes**：GPU 宣称支持哪些窗口。
2. §lspci -vv§ 的 **current size** 与 §nvidia-smi -q§ 的 BAR1 Total：平台实际上分配了多少。
3. §nvidia-smi topo -p2p§、CUDA Peer Access 与数据一致性：驱动和上层运行时是否真正可用。

即便第三层返回 OK，也仍需要验证集合通信的数值和性能。

两台机器的 GPU 都报告 `Physical Resizable BAR` 支持 BAR1 最大 **32GB**；但刚开始 `lspci -vv` 的 **current size** 只有 **256MB**。

维护窗口使用带外控制台检查主板 BIOS/UEFI：

| 设置 | 本次方向 | 为什么 |
| --- | --- | --- |
| Above 4G Decoding | 启用 | 为大 PCIe MMIO 预留地址空间 |
| Resizable BAR | 有此选项则启用 | 允许扩大 BAR1 |
| High MMIO 相关配置 | 按主板文档确保充足 | 四卡 32GiB 共约 128GiB 的窗口地址需求 |
| CSM / UEFI | 与既有 UEFI 启动保持一致 | 防止变更造成不能启动 |
| PCIe 链路 | Auto / 合理的 Gen4 配置 | 空闲的 2.5GT/s 不等于负载最大速率 |
| IOMMU | 在可信环境中评估 passthrough | 对 DMA 隔离存在安全权衡 |

我们将节点内核 IOMMU 默认域切为 passthrough，实际日志为：

```text
iommu: Default domain type: Passthrough (set via kernel command line)
```

**安全提醒**：`iommu=pt` 改变 DMA 隔离语义，不应不加评估地应用在多租户/不可信 PCIe 设备环境。也不要为“让拓扑好看”盲目设置 `pcie_acs_override`，它不会改变物理 PCIe 路径。

冷启动后，每张卡的 `nvidia-smi -q` 均显示 `BAR1 Total: 32768 MiB`，`nvidia-smi topo -p2p r`、`w` 中 12 个非对角有向卡对全部为 `OK`。这是能力枚举成功，尚需 CUDA 与 NCCL 数据验证。

## 5. Tensor Core 实测：2 倍的是哪一层？

首次单卡验证时以一张解锁卡对照一张未解锁卡，同条件运行项目的 `mma_bench` 与 `gemm_bench`，数值测试均为 `failures=0`、`max_abs=0`。跨物理卡的对比仍可能受到功耗、时钟和调度差异影响，因此报告实测而非保证。

### MMA 指令吞吐（best TF/s）

![Tensor Core MMA：原始卡与解锁卡对照](/images/posts/rtx4090-unlock/mma.svg)

指令层基准特意保留未变化的 INT8、FP16/FP16 和普通 FP32 FFMA 作为**负对照**。如果所有模式都同幅度上升，就需要怀疑是频率、功耗或测试方法而不只是某个 Tensor 模式的限制被解除。

| 模式 | 原始/对照 | 解锁 | 加速 |
| --- | ---: | ---: | ---: |
| FP16/FP32 | 161.8 | **323.6** | **2.00×** |
| BF16/FP32 | 161.7 | **323.4** | **2.00×** |
| FP8 E4M3 | 323.3 | **644.5** | **1.99×** |
| FP16/FP16 | 323.7 | 323.7 | 1.00× |
| INT8/INT32 | 646.9 | 646.9 | 1.00× |
| FP32 FFMA | 77.3 | 77.4 | 约 1.00× |

**不是所有 Tensor 模式、所有精度都翻倍**。

### cuBLAS GEMM 中位吞吐（TF/s）

![cuBLAS GEMM 中位吞吐对比](/images/posts/rtx4090-unlock/gemm.svg)

GEMM 的实际吞吐包含 Kernel 实现、访存、矩阵维度与调度成本。特别是 BF16/FP32 4096² 只改善约 0.6%，提示这类任务可能不是被解锁的指令速率控制。

| 算例 | 原始/对照 | 解锁 | 改善 |
| --- | ---: | ---: | ---: |
| FP16/FP32，1024² | 110.38 | 126.72 | +14.8% |
| FP16/FP32，4096² | 163.09 | 199.54 | **+22.3%** |
| BF16/FP32，4096² | 152.31 | 153.22 | +0.6% |
| FP16/FP32，8192² | 171.07 | 189.02 | +10.5% |
| BF16/FP32，8192² | 167.42 | 187.42 | +11.9% |

这里已经能看到：指令性能即便 2×，应用 GEMM 的吞吐提升也会受算子实现、访存和调度影响。

## 6. CUDA Peer Copy 实测：24 条方向全部 PASS

每个方向使用 64MiB 数据，重复 20 次；4 张 GPU 每节点共有 12 条有向卡对，两台共 **24 条**。所有卡对的 `peer_access=1`、`correct=1`，结果均为：

```text
RESULT: PASS; errors=0
```

挑选代表性路径（GiB/s）：

| 路径 | compute-a | compute-b |
| --- | ---: | ---: |
| GPU0 → GPU1，同 NUMA | 21.321 | 21.334 |
| GPU2 → GPU3，同 NUMA | 21.187 | 21.194 |
| GPU0 → GPU2，跨 NUMA | 16.040 | 16.027 |
| GPU2 → GPU0，跨 NUMA | 17.311 | 17.298 |

跨 NUMA 通常比同 NUMA 慢，这是物理主机桥/处理器互连拓扑的结果。此表是**单向拷贝带宽**，不是 NCCL 的 `busbw`。

## 7. NCCL：为什么四卡结果没有“暴涨”？

从 [NVIDIA NCCL](https://github.com/NVIDIA/nccl) 源码构建 **2.31.2**，测试采用 [nccl-tests](https://github.com/NVIDIA/nccl-tests) **2.21.1**。计算节点未安装 NCCL RPM，官方 CUDA 仓库中也未找到该 Fedora 版本的 `libnccl-devel`，故由联网端取源码并在 Fedora 构建镜像编译。

当时 `nvcc 13.3` 报：

```text
#error -- unsupported GNU version!
gcc versions later than 15 are not supported!
```

Fedora 44 默认 GCC 16，因此改为安装兼容 `gcc15`/`gcc15-c++`，明确传入 `CC=/usr/bin/gcc-15`、`CXX=/usr/bin/g++-15`，没有直接绕过 `nvcc` 的编译器版本检查。期间还曾因 `cd` 失败而误在 NCCL 源码目录再次 `make`，耗时很长；之后通过核对工作目录，只构建测试程序。

### 7.1 单机四卡，开启与关闭 P2P

首轮 8–512MiB AllReduce，平均 `busbw`（GB/s）：

| 节点 | P2P Enabled | P2P Disabled |
| --- | ---: | ---: |
| compute-a | 18.4579 | 18.4074 |
| compute-b | 18.0705 | 18.3846 |

**这组数字并未证明四卡 AllReduce 因 P2P 大幅加速。** 某些配置甚至更慢，属于需要进一步分解 NCCL 运行策略的问题。

但开启详细初始化日志并设置 `NCCL_P2P_LEVEL=SYS` 后，两台的 NCCL 明确报告：

```text
Check P2P Type isAllDirectP2p 1 directMode 1 isAllCudaP2p 1
Channel ... via P2P/direct pointer
```

证明该诊断实验里 NCCL 确实使用了 CUDA 直接 P2P 路径。`SYS` 是允许访问的最远拓扑级别，不是“一定选最快路由”的开关。

### 7.2 两卡的 P2P 收益很明显

在其中一台服务器上，32–256MiB 双卡 AllReduce 平均 `busbw`（GB/s）：

| 卡对 | P2P 开 | P2P 关 | 提升 |
| --- | ---: | ---: | ---: |
| GPU0–GPU1 | 19.0955 | 15.8076 | **+20.8%** |
| GPU2–GPU3 | 19.1395 | 15.7801 | **+21.3%** |
| GPU0–GPU2（跨 NUMA） | 17.2592 | 13.7892 | **+25.2%** |

全部 `#wrong=0`。跨 NUMA 禁用 P2P 的数据存在一次明显低值，`+25.2%` 可能有异常值放大效应，不能无条件外推。

### 7.3 三轮复测：AUTO 比强制 PHB / SYS 更快

![四卡 NCCL AllReduce：AUTO、PHB、SYS 的重复实验](/images/posts/rtx4090-unlock/nccl.svg)

为何不把双卡的 21% 改善直接推广到四卡？因为 AllReduce 不是一次独立 Peer Copy，而是涉及同步、分块、通信环/树与不同拓扑边的集合操作；最慢链路、通道选择和 CPU Socket 间路径会影响总吞吐。强制扩大 P2P 允许距离，可能令 NCCL 不再采用其默认认为更好的组合。

因此我们固定消息大小与迭代数量，对 AUTO、PHB、SYS 各重复 3 次，并以所有轮次的平均 §busbw§ 讨论性能，而不是挑最快的一条数据。

32–256MiB 单机四卡 AllReduce，每策略 3 次，平均 `busbw`（GB/s）：

| NCCL 策略 | compute-a | compute-b |
| --- | ---: | ---: |
| **AUTO（不设置 P2P_LEVEL）** | **18.4943** | **18.4803** |
| PHB | 16.9091 | 16.9329 |
| SYS | 16.5636 | 16.5749 |

AUTO 比 PHB 快约 **9.4%**，比 SYS 快约 **11.6%**。全部 18 次测试正常结束，`#wrong` 均为 0，没有越界数据错误。AUTO 由 NCCL 自己选择拓扑和通道，**不要把调试时的 `NCCL_P2P_LEVEL=SYS` 永久写到集群全局环境**。但 AUTO 的详细底层路径仍需要针对具体任务诊断，不能仅凭总线带宽反推所有环节的传输方式。

推荐维持默认：

```bash
unset NCCL_P2P_LEVEL
unset NCCL_P2P_DISABLE
unset NCCL_ALGO
unset NCCL_PROTO
```

### 7.4 对性能指标的解释

NCCL 的 §algbw§ 是算法层有效带宽；§busbw§ 是依据通信原语和卡数折算的总线带宽，**不能直接与单向 CUDA Peer Copy 的 GiB/s 作数值等价比较**。另外本次 §GB/s§ 为 NCCL 的十进制单位，Peer Copy 的 §GiB/s§ 为二进制单位，单位差异也是比较时必须说明的。

实验证据分四级：

| 证据 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| §gpu_unlock: unlocked§ | 固件阶段报告解锁 | 所有应用都加速 |
| §topo -p2p OK§ | 运行时宣称可访问 | 数据必定正确 |
| 24/24 Peer Copy PASS | 指定大小/方向数据正确 | NCCL 必选此路径 |
| §via P2P/direct pointer§ + AllReduce | NCCL 此次使用直接指针、数值通过 | 全部模型训练都提升 |

这四级证据也是本文最希望提供的一套评估方法。

## 8. 这次最有用的踩坑清单

1. **Warewulf bind 的 `not a directory`：** 目的路径可能在挂载失败时留下 0 字节普通文件。先 `stat` 确认、清理该特定空文件，再创建目录；不要盲目递归删除。
2. **交互 Shell 意外退出：** 直接启用 `set -euo pipefail`，后续命令出错可能退出当前交互 Shell。长步骤使用单独 `bash` 脚本并保留日志。
3. **`DKMS added` ≠ `installed`：** 不要仅以 RPM 安装完成判断 NVIDIA 内核模块存在；看 `modinfo -k` 与 `vermagic`。
4. **dracut 的 `No /dev/log` 提示：** 需结合真实退出码、`lsinitrd -m` 的 `wwinit`、固件和模块检查判断，不应只看日志中 `[E]` 就宣告失败。
5. **NFS root_squash：** 无状态节点直接以 root 写 NFS 可能遇到 Permission denied；测试结果先写节点本地，再从管理端主动拉取归档。
6. **错误地在 NCCL 源码里运行 make：** 由于上一条 `cd` 失败，后续 `make` 启动了完整 NCCL 构建。关键目录切换用 `cd ... || exit 1` 防止误操作。
7. **INFO 日志污染性能输出：** NCCL `INFO` 可能与性能表格输出交错；性能跑分用 `WARN`，路径诊断单独保留 `INFO` 日志。

## 9. 运维结论和下一步

### 9.1 可回滚比“一次成功”重要

维护期应先通过 Slurm drain 保护作业，保留原始 NVIDIA 610 镜像、仅 Tensor 解锁的已知可用镜像，以及 615+P2P 独立测试镜像。任何 boot 参数（尤其 IOMMU、GPU BDF）须在节点级设置并检查实际 §/proc/cmdline§；不要改共享 GPU profile 后让另一台不相关节点意外继承。

除冷启动恢复，还应把补丁源码提交哈希、目标内核 ABI、最终 §modinfo filename/version/vermagic§、NCCL 测试日志等收进可追踪的维护记录。

这次已实证完成：两个节点的 Tensor 解锁、32GiB BAR1、24/24 CUDA P2P 有向传输校验、NCCL 直接指针路径验证及短时 AllReduce 数值正确性。**双卡 P2P 收益明显，四卡应保留 NCCL AUTO。**

尚未完成的是长时间满载老化测试、跨节点八卡 NCCL 以及**把手工补丁模块纳入正式 DKMS/RPM** 的持久化管理。特别是驱动/内核自动升级可能覆盖 `updates/dkms` 的补丁模块；应固定镜像版本、保留可冷启动回滚的原镜像，并在实际科研或训练负载中继续记录温度、功耗、Xid 和 PCIe AER。

这次最大的经验不是某个跑分数字，而是要把四件事分别验收：**驱动初始化成功、CUDA 能直接通信、NCCL 实际选择直接通信、应用端最终得到性能收益**。没有这四层分离，就很容易把“能用”误写成“加速已经完成”。

## 参考

- [JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada)
- [Duanyll/open-gpu-kernel-modules](https://github.com/Duanyll/open-gpu-kernel-modules)
- [RTX 4090 48G P2P 与动态 BAR1 技术文章](https://duanyll.com/2026/7/13/4090-48G-P2P/)
- [NVIDIA NCCL](https://github.com/NVIDIA/nccl)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [Warewulf](https://warewulf.org/)
