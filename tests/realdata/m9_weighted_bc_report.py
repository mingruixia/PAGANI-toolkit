"""Summarize paired real-subject M9 measurements and verify the node archives."""
import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(folder):
    rows = []
    for graph_file in sorted(folder.glob("subject-*/density-*/graph.json")):
        case = graph_file.parent
        graph = load(graph_file)
        pairs = []
        for archive in sorted(case.glob("nodes-*.npz")):
            repeat = archive.stem.split("-")[1]
            cpu, gpu = [load(case / f"{b}-{repeat}.json") for b in ("cpu","gpu")]
            with np.load(archive) as a:
                assert len(a["cpu"]) == graph["nodes"]
                assert np.array_equal(a["node_index"], np.arange(graph["nodes"]))
                err = np.abs(a["cpu"]-a["gpu"])
                assert np.array_equal(err, a["absolute_error"])
                np.testing.assert_allclose(a["cpu"],a["gpu"],rtol=1e-6,atol=1e-8)
                pairs.append(dict(repeat=int(repeat),cpu_s=cpu["seconds"],gpu_s=gpu["seconds"],
                    speedup=cpu["seconds"]/gpu["seconds"],max_abs=float(err.max()),
                    max_rel_nonzero_cpu=float(np.nanmax(a["relative_error"])),different_nodes=int(np.count_nonzero(err)),
                    buffer_MiB=gpu["buffer_bytes"]/2**20,board_start_GiB=gpu["board_start_bytes"]/2**30,
                    board_peak_GiB=gpu["board_peak_bytes"]/2**30,identical_file=cpu["output_sha256"]==gpu["output_sha256"]))
        if pairs:
            row=dict(subject=case.parent.name,density=float(case.name.partition("-")[2]),**graph,pairs=pairs)
            row["cpu_s"]=median(p["cpu_s"] for p in pairs)
            row["gpu_s"]=median(p["gpu_s"] for p in pairs)
            row["speedup"]=row["cpu_s"]/row["gpu_s"]
            rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--holdout",type=Path)
    args=parser.parse_args()
    rows=summarize(args.work)
    if not args.report:
        for r in rows:
            print(r["subject"],r["density"],len(r["pairs"]),round(r["cpu_s"],2),round(r["gpu_s"],2),round(r["speedup"],2))
        return
    assert len(rows)==15 and all(len(r["pairs"])==2 for r in rows)
    verification = summarize(args.holdout)
    assert len(verification)==5 and all(r["density"]==0.2 and len(r["pairs"])==2 for r in verification)
    assert not ({r["subject"] for r in rows} & {r["subject"] for r in verification})
    assert load(args.holdout/"hardware.json")==load(args.work/"hardware.json")
    pairs=[p for r in rows+verification for p in r["pairs"]]
    hw=load(args.work/"hardware.json")
    summary=[]
    for d in sorted({r["density"] for r in rows}):
        group=[r for r in rows if r["density"]==d]
        assert len(group)==3
        reps=[p for r in group for p in r["pairs"]]
        summary.append(dict(density=d,cpu_s=median(r["cpu_s"] for r in group),
            gpu_s=median(r["gpu_s"] for r in group),speedup=median(r["speedup"] for r in group),
            lo=min(r["speedup"] for r in group),hi=max(r["speedup"] for r in group),
            wins=sum(r["speedup"]>1 for r in group),pair_wins=sum(p["speedup"]>1 for p in reps),
            stable_gpu=all(p["speedup"]>=1.10 for r in group+verification if r["density"]==d for p in r["pairs"]),stable_cpu=all(p["speedup"]<=1/1.10 for r in group+verification if r["density"]==d for p in r["pairs"]),
            buffer=max(p["buffer_MiB"] for p in reps),board_peak=max(p["board_peak_GiB"] for p in reps)))
    (args.work/"summary.json").write_text(json.dumps(dict(hardware=hw,by_density=summary,by_subject=rows,holdout=verification),indent=2)+"\n")
    density_list=lambda groups:"、".join(f"{g['density']:g}%" for g in groups) or "无"
    sections=["# M9 真实被试加权介数中心性与 GUI 后端选择", "",
        f"在本机 19,574 节点网络上，**{density_list(g for g in summary if g['stable_gpu'])}** 两档满足 GUI 自动优先 GPU 的建议余量。0.20% 档扩大到八名被试后，GPU 在 {sum(r['speedup']>1 for r in rows+verification if r['density']==0.2)}/8 名上更快，适合留作本机校准档。速度结论只覆盖实测密度点，精确交叉位置随被试而变。","",
        "## 数据与计时","",
        "从历史验证的 8 名被试中，按已排序的网络文件取第 1、4、8 名。每人一次 500 时间点的静息态 fMRI，沿用 fMRIPrep、4 mm 重采样、混杂回归、0.01–0.08 Hz 滤波和组灰质掩模。1% 档直接使用归档的 PAGANI 2.2 Pearson 加权 CSR。其余档在原图的无向边中按权重降序取指定数目，边界同权时按端点编号升序，互反边一起保留。各档保留全部节点及原始浮点权重，图哈希保存在各自的 graph.json。[历史验证](VALIDATION_REPORT.md)","",
        f"CPU 为 {hw['cpu_name']}，固定 32 线程。GPU 为 {hw['gpu']['device']}。每个被试、每档密度独立启动两个 CLI 各两次，第二轮反转执行顺序。每次墙钟时间包含启动、CSR 读入、计算及输出。建网、错误比对和归档在计时外。设备处于通常的桌面运行状态。","",
        "## 配对结果","",
        "先对同一被试两轮时间取中位数，再跨被试取中位数。加速比为 CPU 耗时除以 GPU 耗时，范围表示三名被试的差异。显存列给出该密度六次 GPU 运行的最大值。整卡数据来自 NVML，每次查询后等待 25 ms，包含其他进程和驱动开销，可能漏过很短的峰值。算法缓冲区峰值来自程序内同时存活的 CUDA 分配总量。","",
        "| 密度 | CPU 秒 | GPU 秒 | 加速比中位数（范围） | GPU 胜出被试 | GPU 胜出配对 | 算法 MiB | 整卡 GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in summary:
        sections.append(f"| {g['density']:g}% | {g['cpu_s']:.2f} | {g['gpu_s']:.2f} | {g['speedup']:.2f}（{g['lo']:.2f}–{g['hi']:.2f}） | {g['wins']}/3 | {g['pair_wins']}/6 | {g['buffer']:.2f} | {g['board_peak']:.2f} |")
    sections+=["",f"共比较 {sum(r['nodes']*len(r['pairs']) for r in rows+verification):,} 个节点值（含重复运行）。最大绝对误差 {max(p['max_abs'] for p in pairs):g}，CPU 非零节点上的最大相对误差 {max(p['max_rel_nonzero_cpu'] for p in pairs):g}。{sum(p['identical_file'] for p in pairs)}/{len(pairs)} 对输出文件的 SHA-256 完全相同。CPU 为零的节点以绝对误差验收，记录的相对误差为 NaN。","",
        "## GUI 建议","",
        "1. 在加权介数旁提供「自动 / CPU / GPU」。保留用户手动选择，自动模式记录所选程序、选择原因、图规模和最终耗时。无权介数沿用已有调度。","",
        f"2. 建议初始策略以同类图和同一硬件上的该档所有配对均至少快 10% 作为自动派发条件。满足 GPU 条件的密度点为 **{density_list(g for g in summary if g['stable_gpu'])}**，满足 CPU 条件的点为 **{density_list(g for g in summary if g['stable_cpu'])}**。10% 为耗时比的设计余量，用于减少微小波动导致的切换。初始校准仅适用于本机、19,574 节点、该版本程序和相似图结构。比较边数时允许阈值取整的一条边误差。","",
        "3. 为本机保存校准记录。匹配 CPU、线程数、GPU、驱动、算法版本、节点数、平均度和孤立节点比例。优先使用相近图的实测结果。接近交叉点、拓扑超出已测范围或更换硬件时，初次默认 CPU，可由用户启动一次配对测速，随后复用校准记录。密度不可作为跨节点规模的固定阈值。","",
        "4. 自动选择 GPU 前检测 CUDA 设备、可执行文件及可用显存。依内核的 CSR 固定分配和每批源点工作区估算容量。自动模式若遇到设备或资源错误，可重试一次 CPU，将本次成功输出原子替换到目标路径。数值或输入错误直接报告。强制 GPU 模式应显示具体失败原因。","",
        "5. 当前 Qt 在 mainwindow.cpp 和 pipeline.cpp 中禁用了加权介数。实施时同步更新 Settings/JSON、校验、预设与命令生成。加权 CLI 分别为 exefiles_weighted/CUBC 和 CUBC_GPU，现有命令按目录遍历全部 CSR，混合密度批次应增加单文件入口或按后端分组，确保每个网络只运行一次，成功后再交给 ConvertNII。","",
        "## 复核","",
        "项目内 tests/_work/m9_realdata/summary.json 保存各次耗时、误差、图结构、显存和设备信息。每个 subject-*/density-*/nodes-*.npz 包含从 0 开始的 CSR 节点序号、CPU 值、GPU 值和三种误差。对应的 .nm、程序日志和输入哈希均已保留。复跑使用 tests/realdata/m9_weighted_bc.py 的 --source、--package、--work 参数，输出到新目录。",""]
    group=[r for r in rows+verification if r["density"]==0.2]
    wins=sum(r["speedup"]>1 for r in verification)
    newpairs=[p for r in verification for p in r["pairs"]]
    extra=["## 0.20% 密度的留出被试验证","",
           f"在其余五名被试上各做两次配对。GPU 在其中 {wins}/5 名被试上更快，逐次获胜 {sum(p['speedup']>1 for p in newpairs)}/10。合计八名被试，GPU 获胜 {sum(r['speedup']>1 for r in group)}/8，加速比范围 {min(r['speedup'] for r in group):.2f}–{max(r['speedup'] for r in group):.2f}。","",
           "| 留出被试序号 | CPU 秒 | GPU 秒 | 加速比 | 最大绝对误差 | 算法 MiB | 整卡 GiB |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    sections.insert(sections.index("## GUI 建议"),
                    "同一节点数下，0.15% 和 0.20% 的平均度分别约为 29.36 和 39.15。CUDA 实现在平均度达到 32 时切换高平均度内核。密度也同步改变孤立节点数和连通分量，因此本次对照无法单独估计内核切换的贡献。\n")
    for r in verification:
        ps=r["pairs"]
        extra.append(f"| {r['subject']} | {r['cpu_s']:.2f} | {r['gpu_s']:.2f} | {r['speedup']:.2f} | {max(p['max_abs'] for p in ps):g} | {max(p['buffer_MiB'] for p in ps):.2f} | {max(p['board_peak_GiB'] for p in ps):.2f} |")
    where=sections.index("## GUI 建议")
    sections[where:where]=extra+[""]
    sections.append("留出被试原始文件位于 tests/_work/m9_realdata_holdout/。两组使用相同的目录结构和计时口径。")
    from math import ceil, floor, log10
    figure=args.report.with_name("M9_REALDATA_SPEEDUP.svg")
    ticks=[g["density"] for g in summary]
    speeds=[r["speedup"] for r in rows+verification]+[1]
    lo,hi=floor(min(speeds)*5)/5,ceil(max(speeds)*5)/5
    def xcoord(d):
        return 80+560*log10(d/ticks[0])/log10(ticks[-1]/ticks[0])
    def ycoord(s):
        return 270-210*(s-lo)/(hi-lo)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 350" font-family="Arial" font-size="13"><title>CPU to GPU wall-time ratio by density and subject</title><rect width="720" height="350" fill="white"/>']
    for v in range(round(lo*5),round(hi*5)+1):
        y=ycoord(v/5)
        svg.append(f'<path d="M80 {y}h560" stroke="#bbb" stroke-dasharray="4 3"/><text x="70" y="{y+4}" text-anchor="end">{v/5:.1f}</text>')
    for d in ticks:
        svg.append(f'<text x="{xcoord(d)}" y="292" text-anchor="middle">{d:g}</text>')
    for k,subject in enumerate(sorted({r["subject"] for r in rows})):
        series=sorted((r for r in rows if r["subject"]==subject),key=lambda r:r["density"])
        color=["#155a91","#bd5000","#217456"][k]
        points=[(xcoord(r["density"]),ycoord(r["speedup"])) for r in series]
        line=" ".join(f"{x:.1f},{y:.1f}" for x,y in points)
        svg.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.extend(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>' for x,y in points)
        svg.append(f'<text x="{80+k*125}" y="27" fill="{color}">{subject}</text>')
    for r in verification:
        x,y=xcoord(r["density"]),ycoord(r["speedup"])
        svg.append(f'<path d="M{x-4} {y-4}l8 8m-8 0l8 -8" stroke="#222" stroke-width="1.8"/>')
    svg+=['<text x="475" y="27">× held-out subjects</text>',
          '<text x="360" y="324" text-anchor="middle">Undirected edge density (%)</text>',
          '<text transform="translate(22 165) rotate(-90)" text-anchor="middle">CPU time / GPU time</text></svg>']
    figure.write_text("\n".join(svg),encoding="utf-8")
    where=sections.index("## GUI 建议")
    sections[where:where]=[f"![各被试 CPU 与 GPU 耗时比]({figure.name})","",
                           "纵轴 1.0 表示两种后端耗时相等。折线连接同一被试的五个实测点。叉号仅表示留出被试在 0.20% 的实测值。",""]
    args.report.write_text("\n".join(sections),encoding="utf-8")


if __name__=="__main__":
    main()
