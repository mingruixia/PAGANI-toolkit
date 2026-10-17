# PAGANI Toolkit（并行图论分析工具包）

基于 GPU 加速的体素水平（高分辨率）脑功能网络构建与图论分析工具。

[English](README.md) · [使用手册（英文）](docs/USER_GUIDE.md) · [更新说明](CHANGELOG.md) · [验证方法](docs/VALIDATION.md) · [验证报告](docs/VALIDATION_REPORT.md)

PAGANI 以每个灰质体素为节点、体素时间序列间经阈值化的相关为边，为每名被试构建脑网络，并计算全局与节点网络指标：

- **网络构建**：Pearson、Spearman、Kendall 相关；相关系数阈值或稀疏度阈值；个体网络与组水平网络（r 平均或 Fisher-z 平均）；无权与加权网络。
- **全局指标**：特征路径长度、聚类系数、小世界属性 γ / λ / σ（Maslov–Sneppen 随机网络）、模块化（Louvain、Newman）。
- **节点指标**：度/强度、节点效率、节点聚类系数、介数、特征向量中心度、参与系数，结果输出为 NIfTI 图像。

图形界面可一键运行完整流程；每个步骤也是独立的命令行程序，便于批处理和服务器使用。

## 下载与使用

在 [Releases](https://github.com/mingruixia/PAGANI-toolkit/releases) 页面下载 `PAGANI-Toolkit-<版本>-win64.zip`，解压后直接运行 `PAGANI.exe`，无需安装。

运行环境：64 位 Windows 10/11；NVIDIA 显卡（GTX 10 系列及更新），并安装较新的显卡驱动（支持 CUDA 12，Windows 驱动版本 ≥ 527.41）；无需安装 CUDA 工具包。

基本流程：

1. 将每名被试预处理后的 4D fMRI 图像（未压缩 `.nii`，每人一个文件）放在同一文件夹（工作目录）中；
2. 准备与功能像同尺寸的灰质掩模，设置掩模阈值（掩模值 ≥ 阈值的体素作为节点）；
3. 在界面中选择网络类型、阈值和需要计算的指标，点击 **Run**；右侧日志实时显示各程序输出，出错时自动停止并提示原因；
4. 也可用 **Save Script…** 保存批处理脚本，在其他电脑或服务器上运行；**Load Script…** 可恢复全部设置（兼容 1.x/2.0 保存的脚本）。

命令行参数、输出文件格式、内存需求和常见问题见[使用手册](docs/USER_GUIDE.md)。

## 2.1 版说明

2.1 是 2.0 的维护版本：不增加新功能，命令行参数和文件格式保持兼容，旧脚本可直接使用。全部程序在统一、经过测试的代码基础上重写，修正了代码审查（[docs/CODE_AUDIT.md](docs/CODE_AUDIT.md)）中发现的计算错误和稳定性问题。

**与 2.0 结果可能不同的情况**（详见 [CHANGELOG](CHANGELOG.md)）：

| 情况 | 2.0 的问题 |
|---|---|
| 在 GUI 中按**相关系数阈值**建网 | 阈值始终被当作稀疏度传递（r=0.3 实际按 0.3% 稀疏度建网） |
| GUI 中**加权网络**的 Lp、λ、节点效率 | 实际调用了无权程序 |
| 加权组网络 **Fisher 平均**（`bf`） | 平均后的 z 值没有反变换为 r |
| 高分辨率构建 `ps` 模式 | “Pearson” 网络实际是 Spearman 网络 |
| 所有依赖**随机网络**的指标（λ、γ、σ、随机网络 Q） | 同一秒内生成的随机网络完全相同 |
| **小世界 σ** | λ 与 γ 文件按文件名的几个字符配对，可能错配 |
| **特征向量中心度** | 输出值乘以了特征值，跨网络比较存在尺度偏差；现为单位长度特征向量，特征值另存 `_evalue.txt` |
| 带缩放因子、扩展头或整型数据的 NIfTI | 读取错误 |

可用 `tests/compare_results.py 旧结果目录 新结果目录` 逐文件对比新旧结果。

其他变化：参与系数归一化结果更名为 `_pc_normalized.nm`；不再生成 `BNA_time_log`；所有程序出错时返回非零退出码且不再等待按键；数据超出显存时自动分块处理。

## 从源码编译

需要 CMake ≥ 3.21、MSVC 2022、CUDA 12.x 工具包（GPU 程序）和 Qt 5.15 或 6（GUI），命令见 [README.md](README.md#building-from-source)。只编译 CPU 程序（`-DPAGANI_BUILD_CUDA=OFF`）时也可在 Linux/macOS 上编译和测试。

## 引用

如使用 PAGANI，请引用：

Du H, Xia M, Zhao K, Liao X, Yang H, Wang Y, He Y. PAGANI Toolkit: Parallel graph-theoretical analysis package for brain network big data. *Human Brain Mapping* 2018. https://doi.org/10.1002/hbm.23996
