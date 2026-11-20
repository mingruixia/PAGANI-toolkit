# PAGANI Toolkit 功能开发计划（v2.3 / v3.0）

> 2.1、2.2 遵循“只修正、不加功能”（见 [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md)）。自本计划起开始增加功能，但以下约束保持不变：
>
> - **向后兼容**：现有程序名、命令行参数、输出文件格式与命名不变，旧脚本（含 1.x/2.0/2.1/2.2 保存的脚本）可直接运行；新功能只以新程序、新参数（可选、放在末尾）或新输出文件的形式加入。
> - **现有结果不变**：新功能不得改变已有指标的数值；如确需改变，单独提出并在 CHANGELOG 中逐项说明。
> - **先验证后发布**：每个新指标都要先与独立的参考实现（bctpy / networkx / numpy）在合成数据上逐项一致，再在真实数据上验证。

---

## 1. 研发目标

| 编号 | 目标 | 版本 | 验收标准 |
|---|---|---|---|
| **F1 压缩输入** | 所有读 NIfTI 的程序直接读取 `.nii.gz`（BOLD 与掩模） | 2.3 | 同一数据的 `.nii` 与 `.nii.gz` 输入得到**逐字节相同**的网络与输出文件，且输出文件名相同；大文件按卷流式解压，内存占用不超过 `.nii` 输入 |
| **F2 Linux 命令行版** | 全部命令行程序（含 GPU 程序）的 Linux 发行包 | 2.3 | CI 在 Linux 上编译 GPU 程序并跑通 CPU 测试；发行包 `PAGANI-Toolkit-<版本>-linux-x64.tar.gz` 在 glibc ≥ 2.28 的发行版（RHEL/Rocky 8、Ubuntu 20.04 及更新）上无需安装 CUDA 即可运行；GPU 测试套件在 Linux + NVIDIA GPU 上全部通过 |
| **F3 局部效率** | 新指标：节点局部效率与全局平均局部效率（无权/加权），可选随机网络归一化 | 3.0 | 与 networkx `local_efficiency`（无权）及 BCT `efficiency_wei`（加权；参考值用 MATLAB 版 BCT 生成并存入测试数据）一致，相对误差 < 1e-5；19,574 节点、1 % 稀疏度的网络单个计算 < 1 分钟 |
| **F4 加权介数** | 新指标：加权网络介数中心度（距离 = 1/权重） | 3.0 | 与 bctpy `betweenness_wei` 一致（相对误差 < 1e-4）；GPU 版在上述网络上 < 2 分钟 |
| **F5 同配系数与富人俱乐部** | 新指标：度/强度同配系数；富人俱乐部系数曲线及其随机网络归一化 | 3.0 | 与 bctpy `assortativity_bin/wei`、`rich_club_bu/wu` 一致；随机网络与 λ、γ、σ 使用同一组（同一 `PAGANI_SEED` 下可复现） |
| **F6 界面与文档** | 新指标进入 GUI 与脚本；手册、CHANGELOG、验证报告更新 | 2.3 / 3.0 | 新选项可在脚本中完整往返，旧脚本读入后行为不变；中英文文档同步 |

**本计划不包含**（需求明确后另行立项，见第 5 节）：负相关网络、高分辨率加权 Lp 的 GPU 新算法、Python 接口、Linux GUI、BOLD 去噪等预处理功能、组间统计与可视化。

---

## 2. 版本与里程碑

| 阶段 | 内容 | 规模 | 产出 |
|---|---|---|---|
| **M6 压缩输入** | F1（含 GUI 中的文件选择与提示） | 小 | `v2.3.0` 的一部分 |
| **M7 Linux 命令行版** | F2 | 中 | Linux 发行包；与 M6 一起发布 **`v2.3.0`** |
| **M8 局部效率** | F3 无权与加权 CPU 实现 | 中 | 两个命令行版本及合成数据验证已完成 |
| **M9 加权介数** | F4（CPU 参考实现 + GPU 版） | 中–大 | CPU 和 CUDA 命令行均已实现，通过逐节点对照 |
| **M10 同配与富人俱乐部** | F5 | 小–中 | 新程序 + 测试 |
| **M11 集成与发布** | F6：GUI、脚本、文档、真实数据验证 | 中 | **`v3.0.0`** |

2.3 不增加分析指标、不改变任何结果，因此仍用 2.x 版本号；3.0 增加新指标。每个里程碑在 `develop` 上完成并通过 CI，GPU 与真实数据验证由你在本地 GPU 电脑上用 CI 生成的 `dev-build` 包运行（Linux 版可通过 Docker Desktop 的 WSL2 GPU 支持运行）。

---

## 3. 技术方案

### 3.1 M6：`.nii.gz` 输入

- **依赖**：zlib 1.3.1，通过 CMake `FetchContent` 从源码静态编译，Windows 与 Linux 都不依赖系统库，发行包仍为免安装。
- **读取**：`nifti.cpp` 内部改为“字节流”抽象（普通文件 / gzip 流）。gzip 流只能顺序读取，现有的 `read_nifti_masked` 本来就是按卷顺序读取，跳到 `vox_offset` 只需向前跳过，因此流式读取、内存占用不变。头信息校验、数据类型、缩放、字节序处理全部复用。
- **文件发现**：建网程序在工作目录中同时识别 `.nii` 与 `.nii.gz`（不区分大小写）；同一被试同时存在两种文件（如 `sub01.nii` 与 `sub01.nii.gz`）时报错并列出冲突文件，而不是任选其一。
- **输出命名**：被试名去掉完整的 `.nii.gz` 后缀，与 `.nii` 输入时完全相同（`path_without_extension` 增加对双扩展名的处理，并补测试）。
- **掩模**：`.nii.gz` 掩模同样支持。`ConvertNII` 的输出仍为未压缩 `.nii`（保持兼容）。
- **GUI**：掩模文件选择对话框与提示文字接受 `.nii.gz`（识别到的被试文件数由建网程序在运行日志中给出）。
- **测试**：pytest 用 nibabel 生成 `.nii.gz`（含扩展头、缩放因子、int16），比较与 `.nii` 输入的输出逐字节相同；损坏的 gzip 文件给出明确错误。
- **说明**：fMRIPrep 的 `desc-preproc_bold` 未做去噪（头动回归、滤波等），不能直接用于建网；手册中给出建议流程（如 XCP-D 的 `desc-denoised_bold.nii.gz` 输出，或 `tests/realdata/prepare_fmriprep.py` 的做法），PAGANI 本身不做预处理。

### 3.2 M7：Linux 命令行版

- **编译环境**：在 Rocky Linux 8 容器（glibc 2.28）中用 CUDA 12.x 编译，使发行包能在较旧的集群系统上运行；CUDA runtime 与 C++ 运行库静态链接（`-static-libstdc++ -static-libgcc`），只依赖 NVIDIA 驱动。
- **代码**：CPU 工具已在 Linux CI 中编译和测试；需检查并补齐 GPU 工具与打包脚本中的 Windows 假设（路径、`.exe` 后缀、`packaging/make_package.ps1` 增加对应的 shell 版本）。
- **CI**：新增 Linux CUDA 作业：编译全部工具、运行 CPU 测试、生成 tar.gz；打标签时与 Windows zip 一起发布。
- **发行包**：`exefiles/`、`exefiles_weighted/` 目录结构与 Windows 一致（程序名无 `.exe`），附示例 shell 脚本（与 GUI 保存的 `.bat` 脚本步骤对应）。
- **GUI**：不在本阶段范围内。

### 3.3 M8：局部效率（LocalEfficiency）

- **无权版**：先构建每个节点的邻居诱导子图，排除中心节点。局部效率是子图内全部有序不同节点对的最短路径倒数均值。不可达对贡献 0，度小于 2 的节点取 0。全局结果对所有节点取平均。定义参见 [NetworkX local_efficiency](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.efficiency_measures.local_efficiency.html)。
- **无权 CPU 算法**：邻居诱导子图使用稀疏邻接表，每个源点用 BFS 计算最短路，按节点并行。每个线程复用工作区，全局均值按原节点顺序累计。运行时间取决于邻居子图的规模和连通结构。
- **命令和输出**：`LocalEfficiency dir_for_csr num_of_random_networks flags`。用 `g` 写均值和可选随机网络均值，用 `n` 写节点值，用 `r` 写随机网络归一化值。输出分别为 `_leff.txt`、`_leff.nm` 和 `_leff_norm.txt`，详细约定见 [使用手册](USER_GUIDE.md)。给定 `PAGANI_SEED` 时，复用无权 λ 和 γ 所用的随机网络生成序列。
- **加权版**：采用 MATLAB BCT 的修订版 `efficiency_wei(W, 2)`。邻居子图的边长为 `w^(-1/3)`，用这些长度求最短路 `d*`。每个有序邻居对 j、h 的分子贡献为 `(w_ij w_ih)^(1/3) / d*`。以邻居对数求均值，不可达对贡献 0，度小于 2 的节点取 0。CPU 后端用稀疏邻居子图与 Dijkstra。默认使用 CSR 中的原始权重。加权目录中的命令接受可选末尾参数 `--normalize-weights`，按网络最大权重缩放。随机网络保留边权分布，使用相同缩放和种子约定。参见 [MATLAB BCT 实现](https://raw.githubusercontent.com/brainlife/BCT/9465343d33bfdfd22fd0a6f0c10235f544f0e427/BCT/2019_03_03_BCT/efficiency_wei.m)。

#### 2026-11-20 无权版验证记录

Windows 11 的 Release CPU 构建通过，使用 MSVC 19.34 和 CMake 3.24。回归套件 75 项通过，24 项按配置跳过。跳过项需要 GPU、启用大规模测试或构建 Qt GUI。本轮包含 9 项 LocalEfficiency 命令测试，覆盖手算例子、NetworkX 节点和全局对照、归一化、固定种子跨线程复现及异常输入。

| 19,574 节点的无权合成图 | 无向边数 | 实际密度 | 命令耗时中位数 | 平均局部效率 |
|---|---:|---:|---:|---:|
| 随机图，固定种子 44 | 1,915,610 | 1.0000% | 0.90 s | 0.113837 |
| 每个节点与环上左右各 98 个节点相连 | 1,918,252 | 1.0014% | 1.95 s | 0.873068 |

每种图执行 3 次，设置 `PAGANI_THREADS=32`。命令同时读取 CSR 并写出节点和全局指标，随机网络数为 0。随机图抽查 5 个节点的邻居子图，环形图利用旋转对称性核对全部节点。两项均与 NetworkX 对照通过。本记录覆盖 Windows CPU 的合成数据，真实数据性能和 Linux 发行包验证留在后续里程碑。


#### 2026-11-20 加权版验证记录

使用 MATLAB R2026a 直接运行固定版本的 BCT 原函数，对四组小图分别计算原始权重与最大权重缩放后的参考结果。PAGANI 的节点值和全局均值通过全部 8 组对照。独立 SciPy 全对最短路测试进一步核对了随机加权小图。全量 Windows CPU 回归 91 项通过、24 项按配置跳过，其中局部效率专项 25 项全部通过。CMake 安装验证了无权及加权程序都进入对应目录。[MATLAB 参考数据和源码版本](../tests/fixtures/bct_weighted_local_efficiency.json)

| 19,574 节点的加权合成图 | 实际密度 | 命令耗时中位数 | 平均局部效率 |
|---|---:|---:|---:|
| 随机图 | 1.0000% | 2.90 s | 0.056138 |
| 环形图，每节点连接左右各 98 个节点 | 1.0014% | 17.28 s | 0.452335 |

沿用无权基准的两种拓扑，用种子 65 从 [0.1, 1) 均匀生成边权。每种运行 3 次，使用 32 个 CPU 线程和原始权重输出节点值及全局均值。耗时含 CSR 读取和结果写出，随机网络数为 0。两种图各抽查 5 个节点，与独立最短路参考一致。真实数据及 GUI 集成列入 M11。


### 3.4 M9：加权介数中心度

- **定义**。以边长 1/权重计算最短路径，按有序节点对累加未归一化的节点介数。每对可达节点按其等长最短路径数分摊，端点不计入路径介数。不可达节点对和孤立节点贡献为零。已固定对照 [MATLAB BCT betweenness_wei](https://raw.githubusercontent.com/brainlife/BCT/9465343d33bfdfd22fd0a6f0c10235f544f0e427/BCT/2019_03_03_BCT/betweenness_wei.m) 源码版本。
- **CPU 参考实现**。多线程逐源运行双精度 Dijkstra，距离确定后沿严格递增的距离顺序统计路径数，再逆序累积依赖值。两次遍历共用 1e-12 的相对容差，路径数在对数域累加。加权 CSR 通过 `exefiles_weighted/CUBC(.exe) dir` 计算，输出 `<network>_bc.nm`。GUI 集成列入 M11。
- **CPU 验收**。MATLAB R2026a 上运行 BCT 得到五组[逐节点参考值](../tests/fixtures/bct_weighted_betweenness.json)。加权菱形结果为 `[1, 1, 1, 1]`，等长直连和两跳结果为 `[0, 1, 0]`，断连的菱形、三点链和孤立节点组合结果为 `[1, 1, 1, 1, 0, 2, 0, 0]`，均与手算一致。BCT 使用精确浮点比较，PAGANI 另外验收相对容差的两侧。专项用例 18 项全部通过，CPU 回归 109 项通过、24 项跳过。Windows Release 构建及安装路径也已验证。本地日志和 JUnit XML 保存在 `tests/_work/m9_cpu/`。
- 加权 GPU 命令为 `exefiles_weighted/CUBC_GPU(.exe) dir`。GPU 按源点分批计算双精度距离，距离收敛后在最短路径有向无环图上统计等长路径并逆序累积介数。固定顺序合并不同源点的贡献。与 CPU 共用 CSR 输入、边长和结果格式。
- 本机使用项目内的 Windows CI 同版 CUDA 12.6.2 工具链，在 RTX 3080 Ti 上执行全部 18 项 CPU/GPU 配对测试。另用 K64,64 逐节点核验 64 条等长路径、断连分量及显存受限时的分批计算。大图使用 19,574 个节点和约 1% 密度。逐项误差、32 线程 CPU 与 GPU 耗时、算法缓冲区及整卡采样峰值见 [M9 GPU 验收报告](M9_WEIGHTED_BETWEENNESS_GPU.md)。
- 真实被试在五个密度点完成配对测试，另用五名留出被试复核 0.20% 档。GUI 的默认后端和本机校准条件见[M9 真实数据与后端选择](M9_REALDATA_BACKEND.md)。
- Qt 界面已接入加权介数、逐网络自动派发和显式校准。真实混合批次、脚本与控件的验收见 [M9 GUI 集成报告](M9_GUI_BACKEND.md)。

### 3.5 M10：同配系数与富人俱乐部

- **同配系数**：Newman 2002；无权按度、加权按强度（BCT `assortativity_bin` / `assortativity_wei`，flag 0）。程序 `Assortativity(.exe) dir_for_csr`，输出 `<network>_assort.txt`。
- **富人俱乐部**：无权 φ(k)（BCT `rich_club_bu`），加权 φ_w(k)（Opsahl et al. 2008，BCT `rich_club_wu`），k = 1…k_max；归一化 φ_norm(k) = φ(k) / 平均 φ_rand(k)，随机网络与其他指标同一组。程序 `RichClub(.exe) dir_for_csr num_of_random_networks [k_max]`，输出 `<network>_richclub.txt`（列：k、φ、平均 φ_rand、φ_norm、满足条件的节点数）。
- 计算量小，CPU 实现即可。

### 3.6 M11：集成、文档与发布

- **GUI**：在“Network metrics”中加入局部效率、同配系数、富人俱乐部和加权 BC 选项；参数校验（随机网络数、k_max）；新增步骤写入运行日志与保存的脚本。脚本中的设置块增加新键，旧脚本读入时新选项默认关闭。
- **文档**：手册增加新程序的参数、输出文件、内存与运行时间估算和定义说明（含公式与参考文献）；README（中英文）、CHANGELOG 列出新增功能；验证报告增加新指标章节。
- **验证**：合成数据与 bctpy/networkx 逐项比对（CI）；GPU 测试套件（本地）；真实数据（与 2.x 验证相同的 8 名被试）计算全部新指标，检查数值范围与运行时间。

---

## 4. 验证方案

1. **参考实现**：在 `tests/pagani_ref.py` 中加入局部效率（无权用 networkx，加权按公式独立实现）、加权介数（bctpy 公式的 numpy 实现）、同配系数、富人俱乐部；合成网络覆盖不连通、孤立节点、度为 1 的节点、等权重并列路径。
2. **兼容性**：现有测试全部保持通过；新增测试确认 `.nii.gz` 与 `.nii` 输出逐字节相同、旧脚本读入后生成的命令不变。
3. **GPU 与 Linux**：`tests/run_validation.bat`（Windows）及对应的 shell 脚本（Linux / Docker）运行 GPU 测试套件。
4. **真实数据**：沿用 `tests/realdata/` 的流程与数据，报告新指标的结果与运行时间，并确认已有指标与 2.2 逐文件一致。

---

## 5. 后续候选（本计划之外）

| 候选 | 说明 | 前提 |
|---|---|---|
| 负相关网络 | 建网时可选取 \|r\| 或分别输出正、负网络 | 影响所有加权下游指标的解释，需先确定方法学约定 |
| 高分辨率加权 Lp | GPU 单源最短路替代 Floyd–Warshall（内存 O(边数)），用于 6 万节点以上的加权网络 | 有实际高分辨率加权分析需求；可复用 M9 的 GPU 最短路代码 |
| Python 接口 | pybind11 封装核心库，pip 安装 | 3.0 稳定后；需维护多平台 wheel |
| Linux GUI | Qt GUI 的 Linux 版 | M7 完成后视需求 |
| NIfTI-2 输入 | 超大图像的 NIfTI-2 格式 | 有用户需求时 |

---

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| 加权局部效率、加权富人俱乐部存在多种定义，结果与其他软件不可比 | 实现前确认定义；手册写明公式与对应的 BCT 函数；必要时提供定义选项 |
| 加权介数在大网络上耗时过长 | 先完成 CPU 参考实现并实测；GPU 版不达标时在手册中给出运行时间估算与网络规模建议 |
| 加权最短路的浮点并列判定导致与参考实现的细微差异 | 统一容差，测试中构造精确并列与近似并列两类情形 |
| Linux 发行包与集群环境（glibc、驱动版本）不兼容 | 在旧 glibc 容器中编译、静态链接运行库；Docker 中测试多个发行版 |
| 新功能引入对已有结果的回归 | 现有测试与真实数据逐文件比对作为每个里程碑的合并条件 |
| 本地无法编译 | 构建与 CPU 测试在 GitHub Actions 中进行；GPU 验证使用 CI 生成的 `dev-build` 包 |

---

## 7. 待确认事项

1. 富人俱乐部是否需要加权版本（Opsahl 2008），还是只做无权版本？
2. Linux 发行包的最低系统版本（本计划默认 glibc 2.28，即 RHEL/Rocky 8）是否满足你所用的集群？
3. 新程序名（`LocalEfficiency`、`Assortativity`、`RichClub`、加权 `CUBC`）及输出文件后缀（`_leff`、`_assort`、`_richclub`）是否合适？
