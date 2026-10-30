# PAGANI Toolkit 维护版研发目标与方案（v2.1）

> 2.3 / 3.0 的功能开发计划见 [`DEVELOPMENT_PLAN_3.md`](DEVELOPMENT_PLAN_3.md)。

> 原则：**只修正、不加功能**。所有分析算法、命令行参数、输出文件格式保持与 2.0 兼容；改动限于修复错误、纠正计算、提升稳定性、改进界面与使用流程、更新构建与文档。
> 详细问题清单见 [`CODE_AUDIT.md`](CODE_AUDIT.md)（问题编号在本文中引用）。

---

## 1. 研发目标

| 编号 | 目标 | 验收标准 |
|---|---|---|
| **T1 计算正确** | 修复全部 P0 问题（G-01/02、C-01~06、M-01/02、S-01、E-01、N-01/02、P-01、CP-01/02、L-01/02） | 每个指标在合成数据上与参考实现一致：相关系数/阈值边集完全一致（容差 1e-5）；Degree、Cp、Lp/效率、EC、PC、BC 与 NetworkX / bctpy 一致（相对误差 < 1e-4）；Q 值不低于参考实现；Maslov 随机网络保持度序列且多次生成互不相同 |
| **T2 稳定可靠** | 修复全部 P1 问题 | 无交互式暂停；失败返回非 0 退出码并给出明确提示；路径含空格/中文正常；内存需求提前估算并给出提示 |
| **T3 可构建、可运行** | 用现代工具链重建全部程序 | CMake + MSVC 2022 + CUDA 12.x + Qt 5.15 一键构建；GitHub Actions 自动编译；在当前主流 NVIDIA 显卡（Pascal 至 Blackwell）上运行 |
| **T4 好用** | 修复 GUI 问题（G-03~11），优化使用流程 | 脚本保存/读取可完整往返；GUI 内显示运行日志与完成状态；参数校验与提示；高分辨率构建（2.0 功能）在 GUI 中可用 |
| **T5 可发布** | 文档与发布 | 中英文 README、使用手册更新、CHANGELOG、示例脚本；GitHub Release `v2.1.0`（免安装 zip + 源码） |

**不在范围内**：新的连接度度量、新的图论指标、新的输入格式（如 `.nii.gz`）、Linux GUI 发行版。若在验证中确有必要（例如 int16 BOLD 数据读取），会单独提出征求意见。

---

## 2. 开发基线

1. 以上游 `BNAplatform-organization/PAGANI-toolkit` 的 `dev` 分支（tag `2.0`，HBM 论文版源码）为基线，将其同步进 fork。
2. 从本地 `PAGANI-toolkit-1.5` 补入仓库中缺失的源码：Kendall 版高分辨率构建（`CuCorMat-sca-kendall`）及 MATLAB 校验脚本。
3. GUI 以 1.4 源码为底，按 2.0 二进制中的界面字符串重建 2.0 新增的“高分辨率构建”选项（CUCorMatHr）。
4. 无权 Maslov 采用旧版（已发布二进制所用）的正确逻辑，结合新版的稀疏数据结构重写。
5. `PAGANI-exefiles` 中无源码的 `LP_modified.exe`、`Cp.exe`(2017-09)、`CUModularity.exe` 不纳入发布（待确认其用途）。

---

## 3. 技术方案

### 3.1 仓库与目录重组

```
PAGANI-toolkit/
├── src/
│   ├── common/        # 共用库：NIfTI 读写、CSR 读写、目录扫描、Maslov 重连、线程、计时、错误处理
│   ├── construct/     # CUCorMat、CUCorMatHr（Pearson/Spearman）、Kendall
│   ├── metrics/       # Degree、Cp、Lp/效率、BC、EC、Louvain、Newman、PC、SmallWorld、ConvertNII
│   └── gui/           # Qt GUI
├── tests/             # 参考实现（Python）、合成数据生成、CPU 单元测试、GPU 验证脚本
├── scripts/           # 示例批处理脚本
├── docs/              # 手册、审查报告、本方案
├── packaging/         # 打包脚本（生成发布 zip）
└── legacy/            # 原 2015/2018 源码树（只读保留，便于对照）
```

- **可执行文件名与命令行参数保持不变**（`exefiles\` 与 `exefiles_weighted\` 两套目录、`.csr/.nm/.txt` 输出格式、文件命名规则不变），旧脚本可直接使用。
- 二进制、pdb、编译日志从版本库中移除，发布物通过 GitHub Releases 分发；不改写已有历史。
- 分支策略：`master`（发布）← `develop`（集成）← `fix/*`、`build/*`、`gui/*` 特性分支；每个里程碑以 PR 合并并推送到 GitHub。

### 3.2 共用库（解决重复代码与系统性问题）

- **NIfTI**：按 `datatype` 判型，支持 uint8/int16/int32/float32/float64 数据与掩模，处理 `vox_offset`、`scl_slope/scl_inter`、字节序；不支持的类型直接报错退出（修 N-01/02、C-05）。
- **CSR**：统一读写与格式校验（文件长度、R 单调、列索引范围），64 位计数防溢出（R-05）。
- **Maslov**：正确输出 CSR、使用 `std::mt19937_64`，每个随机网络独立种子，可通过环境变量固定种子以便复现（M-01/02）。
- **跨平台**：`std::filesystem`、`std::thread` 替换 Windows 专有 API，使 CPU 工具能在 Linux CI 中编译和测试（R-03）。
- **错误处理**：统一的错误输出与非 0 退出码，移除 `pause/cin/getchar`（R-01/02）；GPU 调用全部检查返回值（C-04）。

### 3.3 GPU 代码现代化

- 迁移到 CUDA 12：`cusparseDcsrmv` → `cusparseSpMV`，`cusparseSdense2csr/Snnz` → `cusparseDenseToSparse`，`cusparseScsr2csc` → `cusparseCsr2cscEx2`。
- 编译目标架构 `sm_61;75;86;89;90` + PTX；静态链接 cudart。
- 显存不足时给出所需显存估算，Hr 版本自动启用分块路径。

### 3.4 计算修正要点

| 模块 | 修正 |
|---|---|
| 网络构建 | 统一阈值定义（`r > t`，稀疏度模式精确取前 K 条边并按固定规则处理并列）；加权 Fisher 组平均（C-01）；`ps` 模式分别基于原始数据计算两种系数（C-02）；参数合法性检查（C-06） |
| 小世界属性 | 按完整文件名前缀配对 λ/γ，去除 `char` 下标（S-01） |
| 特征向量中心度 | 输出单位 L2 范数的特征向量（E-01），特征值另存 |
| 参与系数 | 兼容 0/1 起始的模块编号，修正输出文件名（P-01） |
| Lp/Cp | 溢出与参数解析问题（L-01/02、CP-01/02） |

**结果变化说明**：修正后部分指标与 2.0 不同（如 EC 数值尺度、σ）。CHANGELOG 中逐项说明受影响的指标与原因，便于已发表结果的解读。

### 3.5 GUI 与使用流程

- 修复 G-01~11；脚本保存/读取完整往返（含所有选项组合）。
- 可执行文件使用基于 GUI 所在目录的绝对路径，所有路径加引号。
- 通过 `QProcess` 在 GUI 内顺序执行各步骤：实时日志窗口、当前步骤与进度、失败即停止并提示、可中止；同时仍可“保存脚本”供命令行或服务器使用。
- 参数校验与就近提示（阈值范围、随机网络数、小世界依赖项、加权模式下不可用的指标）。
- 恢复 2.0 的“高分辨率构建”选项；界面布局小幅整理，不增减分析功能。

### 3.6 验证方案

1. **参考实现**（Python：numpy / scipy / networkx / bctpy）：对同一合成数据独立计算所有指标。
2. **合成数据**：小规模 4D NIfTI（数百到数千体素，含植入的模块结构）及边界情况（零方差体素、并列相关值、不连通网络、孤立节点、各种数据类型与头信息）。
3. **CPU 工具**：GitHub Actions 中在 Linux/Windows 上自动编译并与参考实现比对。
4. **GPU 工具**：CI 只验证编译；数值验证在你的 Windows + NVIDIA 电脑上运行 `tests\gpu_validate`，生成对比报告。
5. **新旧对照**：同一份真实数据用 2.0 二进制与新版分别计算，差异逐项归因到已修复的问题。

---

## 4. 里程碑

| 阶段 | 内容 | 产出（同步至 GitHub） |
|---|---|---|
| **M0 仓库整备** | 同步上游 `dev`，补入本地源码，目录重组，`.gitignore`，CMake 骨架，CI | `develop` 分支、可编译的骨架 |
| **M1 共用库 + CPU 工具** | NIfTI/CSR/Maslov 共用库，Degree/Cp/PC/Louvain/SmallWorld/ConvertNII/BFS 修正，参考实现与测试 | CPU 工具全部通过测试 |
| **M2 GPU 工具** | CUDA 12 迁移，CorMat/CorMatHr/Kendall/Lp(BFW)/BC/EC/Newman 修正 | CI 编译通过；GPU 验证脚本 |
| **M3 GUI** | GUI 修复、流程优化、2.0 选项重建 | 新 GUI |
| **M4 全流程验证** | 在你的 GPU 电脑上跑验证脚本和真实数据，修复发现的问题 | 验证报告 |
| **M5 发布** | 手册、README、CHANGELOG、打包 | GitHub Release `v2.1.0` |

---

## 5. 风险与应对

| 风险 | 应对 |
|---|---|
| 本环境无 GPU，GPU 代码无法在开发端运行 | CPU 参考实现 + CI 编译检查 + 你本地运行验证脚本 |
| CUDA 12 的 cuBLAS/cuSPARSE 运行库体积大（数百 MB） | 发布包按需拆分 GPU 运行库；或说明从 NVIDIA 安装 |
| 修正后结果与已发表研究不一致 | CHANGELOG 中逐项说明并给出新旧对照 |
| 2.0 GUI 源码缺失 | 依据 1.4 源码与 2.0 二进制界面字符串重建，由你对照旧界面确认 |
