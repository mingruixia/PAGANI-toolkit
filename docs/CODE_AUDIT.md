# PAGANI Toolkit 代码审查报告（v2.0 基线）

审查日期：2026-10-17
审查范围：
- GitHub fork `mingruixia/PAGANI-toolkit`（master = 上游 master，2018-05-28）
- 上游组织仓库 `BNAplatform-organization/PAGANI-toolkit` 的 `dev` 分支 / tag `2.0`（2018-06-01，含 HBM 论文版 `*-20180520` 源码）
- 本地 `PAGANI-toolkit-1.5`、`PAGANI-exefiles`

下文文件路径均相对于上游 `dev` 分支（`src/BNAPlatform-win64-cuda-20180520` 简写为 `U/`，`src/BNAPlatform-weighted-network-win64-cuda-20180520` 简写为 `W/`）。
严重度：**P0** 结果错误（静默算错）；**P1** 崩溃/卡死/数据损坏；**P2** 易用性；**P3** 工程问题。

---

## 一、资源与版本现状

| 来源 | 内容 | 结论 |
|---|---|---|
| fork master（= 上游 master） | 2015-11 版源码树 + 1.4/2.0 GUI 可执行文件 + 二进制 | **缺少**无权重网络的 Lp/Louvain/Newman/PC/小世界等大量源码；不是最新源码 |
| 上游 `dev` / tag `2.0` | `*-20180520` 两套源码树（HBM 论文版）、CUCorMatHr 高分辨率构建、GUI 源码 | **最完整、最新**，应作为开发基线；fork 中没有这个分支 |
| 本地 `PAGANI-toolkit-1.5` | ≈ v1.5（2017）+ 高分辨率构建研究代码（Kendall、Pearson/Spearman 原型、MATLAB 校验脚本） | Kendall 版 `CuCorMat-sca-kendall` 的源码**只在这里** |
| 本地 `PAGANI-exefiles` | 实际使用中的混合二进制集 | `LP_modified.exe`、`Cp.exe`(2017-09)、`CUModularity.exe`(2017-12) 由 `E:/project/BNAPlatform-weighted-win64-cuda7.0-20160808` 编译，**源码不在任何位置** |

关键缺口：
1. **PAGANI-2.0 GUI 源码从未提交**。仓库中 GUI 源码与 1.4 完全一致；2.0 可执行文件相对 1.4 只多了“高分辨率构建（CUCorMatHr）”选项（由二进制字符串比对确认），可在 1.4 源码上重建。
2. **发布的二进制与仓库源码不一致**：无权重工具二进制为 2017-02 编译（旧版稠密 Maslov），而 `dev` 源码中的 Maslov 已改为未完成的新实现（见 M-01）。直接用 `dev` 源码重编会引入回归。
3. 所有二进制基于 CUDA 7.0/7.5/9（VS2012/2013/2017），对 RTX 30/40/50 系列等新显卡的兼容性无法保证；源码中使用的 `cusparseDcsrmv`、`cusparseScsr2csc`、`cusparseSdense2csr`、`cusparseSnnz` 等接口在 CUDA 11/12 中已被移除。

---

## 二、P0：计算结果错误

| ID | 位置 | 问题 | 影响 |
|---|---|---|---|
| **G-01** | `src/GUI/mainwindow.cpp:210, 714`（1.4 与 2.0 二进制均存在） | 阈值类型判断用 `currentText()=="correlation"`，而下拉框文本是 `"Correlation"`（大写），比较永远为假 → **始终传 `s`（稀疏度）** | GUI 用户选择“相关系数阈值 0.3”时，实际按“稀疏度 0.3%”建网，网络密度与预期相差几十到上百倍 |
| **G-02** | `src/GUI/mainwindow.cpp:831–943` | 加权模式下特征路径长度/节点效率调用的是无权重目录 `.\exefiles\CUBFW_Lp.exe` | 加权网络的 Lp、λ、节点效率实际按二值网络计算；且无权重程序按无权 CSR 格式读加权文件 |
| **C-01** | `W/CorMat/CUCorMat/CUCorMat/CorMat.cpp:565` | 加权组平均的 Fisher 反变换判断写成 `argv[3][1]`（掩模阈值参数），应为 `argv[4][1]` | `bf` 模式下平均后的 z 值未反变换回 r：z>1 时程序以 “illegal correlation coefficient” 退出，否则用 z 值做阈值，组水平加权网络错误 |
| **C-02** | `W/CorMat/CUCorMatHr/CorMat-Pspear-dense2csr.cpp:577–620`、`CorMat-Pspear-preprocessing.cu:264` | `ps` 模式先做 Spearman：秩变换和标准化**原地覆盖** `BOLD_t`，随后的 Pearson 轮直接使用已秩变换的数据 | `ps` 输出的 “pearson” 网络实际是 Spearman 网络 |
| C-02b | 同文件 `hashit()` | 只识别 `"ps"`（写成 `"ps"||"ps"`），文档中的 `"sp"` 无返回值（未定义行为） | 参数 `sp` 行为不确定 |
| **M-01** | `U/Cp/Cp/Maslov.cpp:354–488`（同一文件复制在 Lp、BFS_MulCPU、Louvain、Newman 中） | 无权重 `Maslov()` 完成重连、排序后**从不写入 `C_dst`**，调用方拿到的是未初始化内存 | 用 `dev` 源码编译时：所有随机网络指标（随机 Lp/Cp/Q、λ、γ、σ）无效甚至崩溃 |
| **M-02** | 所有 `Maslov.cpp` 的 `Rewire()` | 每次调用 `srand(time(0))`（哈希版本为 `srand(1)`） | 1 秒内生成的多个“随机网络”完全相同（哈希版本永远相同），随机网络数 n 实际退化为 1，σ/λ/γ 的方差被低估 |
| **S-01** | `U/SmallWorldProperty/SmallWorldProperty.cpp:94–125` | λ 与 γ 文件按“文件名第 2–5 个字符 + spa 后 5 个字符”配对；配对下标存放在 `char` 数组中 | 被试名前缀相同（如 `sub001`/`sub0010`）时 λ、γ 错配；文件数 >127 时下标溢出，σ 错配或崩溃 |
| **E-01** | `W/CUEC/weighted-net.cu:186`、`U/CUEC/CUEC/CUEC_unweighted.cu` | 输出的是 `A·u`（范数≈最大特征值 λ），未归一化 | 特征向量中心度被各被试自身的 λ 缩放，跨被试比较存在系统偏差 |
| **N-01** | 所有读 NIfTI 的程序（CorMat、CorMatHr、ConvertNII） | 假设 `vox_offset=352`；忽略 `scl_slope/scl_inter`；用 `bitpix` 判断类型（int32 被当成 float）；BOLD 为 int16/uint8 时只打印错误但继续用**未初始化内存**；掩模仅支持 uint8/float | 带扩展头、缩放因子、整型数据的文件会得到静默错误的网络 |
| N-02 | `ConvertNII.cpp` | 输出头直接拷贝掩模头，只改 datatype/bitpix，未重置 `scl_slope/scl_inter`、`vox_offset`、`cal_min/max` | 掩模带缩放或扩展时输出图像数值/偏移错误 |
| C-03 | `U/.../CorMat.cpp:377,629`；`W/.../CorMat.cpp:361`；CorMatHr 内核 | 阈值判定不一致：无权重 `r > t`，加权 `r ≥ t−1e-6`，Hr `r > t` | 同一阈值下加权与无权网络边集不同 |
| C-04 | `U/.../CorMat.cpp:319` 等 | `CorMat_gpu()` 返回的 CUDA/cuBLAS 错误码被忽略 | 显存不足等失败时使用未初始化的相关矩阵，静默生成垃圾网络 |
| C-05 | CorMat 系列 | 方差为 0 的体素（掩模过宽/信号缺失）产生 NaN，被默默当作 0；组平均仍除以全部被试数 | 结果偏差且无提示 |
| C-06 | `U/.../CorMat.cpp` 参数解析 | `to_average` 只给 `y` 时第二个字符为空，平均循环什么都不累加 | 生成空的组网络，无任何报错 |
| P-01 | `W/PC_CPU/PC_CPU/main.cpp:149` | 假设模块编号为 1..M，`WithinModule[m-1]`；若出现 0 号模块越界写 | 参与系数错误或堆损坏（需与各模块划分程序的编号约定核对） |
| CP-01 | `W/Cp/Cp/Cp_main.cpp:42` | 省略 Cp 类型参数（argc==4）时仍访问 `argv[4]` | 崩溃 |
| CP-02 | `W/Cp/Cp/Cp_2_NoG.cpp` | Onnela 版本对权重原地除以最大权重，修改了输入数组 | 需核实对后续随机网络/输出无副作用 |
| L-01 | `U/Lp/CUBFW_Lp/CUBFW_Lp/CUBFW_Lp_main.cpp:119` | `(numVertices-1)*numVertices` 为 int 乘法，N>46341 溢出 | 算法选择（BFW/BFS）错误，主要影响性能；原网络与随机网络判定条件 `>=`/`>` 不一致 |
| L-02 | `U/Lp/CUBFW_Lp/CUBFW_Lp/CUBFW_Lp.cpp:93` | `x<1 && x>N` 恒假，不连通检测失效 | 不连通网络不提示（数值因 1/inf=0 仍正确） |
| V-01 | BC、Louvain Q、Newman 模块度 | 归一化约定（BC 是否 /2、Q 定义）未与标准实现核对 | 列入验证清单 |

---

## 三、P1：崩溃、卡死与健壮性

| ID | 问题 |
|---|---|
| R-01 | 大量程序在出错或结束时调用 `system("pause")`、`cin>>`、`getchar()`（ConvertNII 结束时必定 pause），批处理流程会卡住 |
| R-02 | 出错时大多 `return 0`，退出码无意义，脚本/GUI 无法判断失败 |
| R-03 | Cp、APSP_BFS 等用 `_beginthread` + `WaitForSingleObject`：`_beginthread` 的句柄在线程结束时自动关闭，存在等待无效句柄/提前读取结果的竞态 |
| R-04 | CUCorMatHr 计数时排除 4 个掩模文件名、填充时只排除 2 个 → 目录中有 `1mm_grey.nii` 时数组越界写 |
| R-05 | 多处 int 溢出：`.cormat` 文件头用 `sizeof(int)` 写 `long long M1`；BC 中 `maxgrid` 计算；`R_type` 为 32 位 |
| R-06 | 每个程序都向当前目录追加 `BNA_time_log`，文件无限增长（本地已达 7 MB） |
| R-07 | GPU 内存/主机内存需求（N×N bool 邻接矩阵、N×N float 距离矩阵）无预估与提示，超限时直接崩溃 |

---

## 四、P2：GUI 与使用流程

| ID | 问题 |
|---|---|
| G-03 | 加权模式 `CheckLp + NodalLp` 分支中 `nodal_metrics=true` 写在 `return` 之后（`mainwindow.cpp:877`），节点效率不会被转换为 NIfTI |
| G-04 | “Load Script” 只识别 `g/n/b` 标志，而 “Save” 生成的是 `gl/gn/gnl/nk/gnk…`；加权 Lp 的 exe 路径与保存时不一致；读入后未同步全局变量（`to_save_cormatrix`、`both_metrics`、`average_fisher`），再次保存会丢失选项 → 脚本无法往返 |
| G-05 | 读入脚本时阈值类型、Cp 类型的下拉框索引映射颠倒（`setCurrentIndex(tokens[6]=="r")` 等） |
| G-06 | 路径不加引号，含空格的路径全部失败；用 `start <脚本>` 执行，路径含空格时被当作窗口标题 |
| G-07 | 依赖当前工作目录的相对路径 `.\exefiles\...`，从快捷方式或其他目录启动 GUI 即失效 |
| G-08 | 缺少参数校验：阈值范围、随机网络数为空时模块度命令缺参数、小世界属性依赖 λ/γ 但不强制勾选 |
| G-09 | 运行在独立控制台窗口，GUI 无进度、无日志、无法得知完成或失败 |
| G-10 | 启动时写入占位文本 `"sdffsssssssss"`；连接到不存在的槽 `on_checkSmallWordProperty_clicked`；Linux 分支路径笔误（CUEC→CUBC、`exefiles_wieghted`） |
| G-11 | 加权模式勾选 BC 会调用不存在的 `exefiles_weighted\CUBC.exe`（目前通过禁用复选框规避） |
| D-01 | 本地 1.5 的 README 是 GRETNA 的；`readme.txt` 与实际参数不符；手册未覆盖 2.0 的高分辨率构建 |

---

## 五、P3：工程问题

- 仓库 305 MB，提交了 exe/dll/pdb/编译日志/`.vs` 数据库/`ipch`；`.gitignore` 不完整。
- 无统一构建系统：VS2012–2017 工程文件、硬编码绝对路径与 CUDA 版本。
- 大量重复代码：`Maslov.cpp` 11 份、`Timer`/`dirent.h`/CSR 读写/目录扫描在每个工具中各写一遍；无权与加权两棵源码树高度重复。
- Windows 专有 API 散落各处（`windows.h`、`_beginthread`、`PathIsDirectory`、`itoa`、`sprintf_s`），无法在 CI 中做自动化测试。
- 无任何自动化测试、无版本号与变更记录。
