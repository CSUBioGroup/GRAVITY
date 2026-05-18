GRAVITY predicts RNA velocity and regulatory rewiring by dynamic regulatory mechanism-enhanced deep learning
==========================================================================================================

GRAVITY 是论文 “GRAVITY predicts RNA velocity and regulatory rewiring by
dynamic regulatory mechanism-enhanced deep learning” 对应的软件实现。它将
RNA velocity 推断与动态基因调控网络建模结合起来，整合未剪接/已剪接 RNA
丰度、细胞嵌入和先验基因调控网络，通过调控网络感知的 attention 结构联合建模
细胞状态转移、基因特异性转录动力学，以及动态调控网络重连。

本仓库提供面向研究使用的 Python 实现。流程先优化细胞层面的 velocity 与未来位置，再细化基因层面的动力学参数，并导出基于 attention 的调控网络摘要用于后续分析。

方法概览
--------
![GRAVITY 方法图](../docs/assets/gravity_method_overview.png)

特性
----
- 基于已剪接/未剪接计数的动态调控网络感知 RNA velocity 推断；
- 两阶段优化：先进行细胞层面的轨迹恢复，再进行基因层面的动力学细化；
- 结合先验 GRN 的 attention 导出，用于调控因子和功能模块分析；
- 支持细胞层轨迹和指定基因的速度可视化。
- 输入和中间计数表参考 cellDancer 的长表数据存储形式，GRAVITY 会进一步转换成两阶段模型使用的内部宽表 `combine.csv`。

安装
----
建议使用 Python 3.10 或 3.11 并创建虚拟环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需 GPU，请先安装对应的 `torch` CUDA 版本，然后执行 `pip install -e .`。例如 CUDA 11.7 环境：

```bash
pip install --index-url https://download.pytorch.org/whl/cu117 "torch==2.0.1+cu117"
pip install -e .
```

演示数据
--------
胰腺内分泌发生 CSV 是 smoke test 和教程使用的真实演示数据。该文件是
cellDancer 胰腺内分泌发生案例教程中链接的预处理输入数据：

```text
https://guangyuwanglab2021.github.io/cellDancer_website/notebooks/case_study_pancreas.html
```

CSV 压缩包下载地址：

```text
https://drive.google.com/file/d/16hV9t66edOgjCmoBuEfekS3ijtL1fYNc/view?usp=sharing
```

下载后保存为：

```text
data/PancreaticEndocrinogenesis_cell_type_u_s.csv
```

仓库中已经包含该演示所需的 mouse 先验网络和胰腺参考 checkpoint。

先验网络
--------
GRAVITY 提供两个物种对应的先验网络压缩包：

```text
prior_data/nichenet_mouse.zip
prior_data/nichenet_human.zip
```

胰腺演示默认使用 `prior_data/nichenet_mouse.zip`。如果分析 human 数据，可在
`PipelineConfig` 中设置 `prior_network="prior_data/nichenet_human.zip"`，
或在运行 smoke test 时设置
`GRAVITY_PRIOR_NET=prior_data/nichenet_human.zip`。

这些压缩包按 CEFCON 描述的 prior-network 处理流程整理。起点是 NicheNet 的
integrated gene interaction network，包含 ligand-receptor、intracellular
signaling 和 gene regulatory interactions。GRAVITY 关注单细胞内部调控，因此
去除了细胞间 ligand-receptor interactions；使用 unweighted integrated
network；将 undirected edges 作为 bidirectional directed edges 处理。human
版本保留 human gene symbols，mouse 版本通过 ENSEMBL one-to-one orthologs
映射并去除有歧义的基因。仓库中以 zipped edge-list CSV 保存，列名为 `from`、
`to` 和 `edge_type`。相关链接：

- NicheNet: https://www.nature.com/articles/s41592-019-0667-5
- CEFCON: https://www.nature.com/articles/s41467-023-44103-3
- cellDancer 数据格式与胰腺演示: https://www.nature.com/articles/s41587-023-01728-5

快速开始（端到端）
-----------------
将演示 CSV 放到 `data/PancreaticEndocrinogenesis_cell_type_u_s.csv` 后运行：

```python
from gravity import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    raw_counts="data/PancreaticEndocrinogenesis_cell_type_u_s.csv",
    workdir="gravity_outputs_pancreas",
    prior_network="prior_data/nichenet_mouse.zip",
    gene_order_path="data/pancreas/reference_checkpoints/pancreas_genes.txt",
    accelerator="gpu",
    devices=1,
    batch_size=16,
    stage1_epochs=6,
    stage2_epochs=4,
    stage1_lr=1e-6,
    stage2_lr=1e-4,
    make_plot=True,
    plot_genes=["GCG", "INS2"],
)
outputs = run_pipeline(cfg)
print(outputs)
```

也可以直接运行脚本：

```bash
python gravity/smoke_test.py
```

预期输出是一个打印出来的路径字典。输出目录中应包含 `combine.csv`、
stage checkpoints、`future_positions.npy`、stage CSV、attention 导出文件，
以及所选基因的 velocity plots。

无监督和对比学习目标对学习率略敏感。参考运行建议从
`stage1_lr < 1e-5` 开始，`stage2_lr` 通常在 `1e-3` 到 `1e-5`
之间调参。

从 h5ad 转换为 CSV
-------------------
GRAVITY 使用和 cellDancer 类似的长表计数存储形式：每一行对应一个
cell-gene pair，并包含剪接/未剪接计数及细胞元信息。若数据源是 AnnData，可先使用
`export_intermediate_from_h5ad` 转成该长表 CSV：

```python
from gravity import export_intermediate_from_h5ad

export_intermediate_from_h5ad(
    input_h5ad="data/postprocessed.h5ad",
    output_csv="data/PancreaticEndocrinogenesis_cell_type_u_s.csv",
    n_top_genes=1000,
    embed_key="X_umap",
    celltype_key="celltype",
)
```
该工具会检查所需的 spliced/unspliced layers，并将嵌入坐标和聚类标签一并写入长表 CSV。

完成后，`workdir` 通常包含：`combine.csv`、`stage1.csv`/`stage1.ckpt`、`future_positions.npy`、`stage2.csv`/`stage2.ckpt`、`attentions/`、`velocity_plots/*.png`（若开启绘图）。

胰腺内分泌发生参考权重位于 `data/pancreas/reference_checkpoints/`：

```text
data/pancreas/reference_checkpoints/pancreas_stage1.ckpt
data/pancreas/reference_checkpoints/pancreas_stage2.ckpt
data/pancreas/reference_checkpoints/pancreas_genes.txt
```

这两个 checkpoint 可以直接作为胰腺 stage-1 和 stage-2 权重使用。对应的参考导出
命名为 `pancreas_stage1_reference.csv` 和 `pancreas_stage2_reference.csv`；
它们是较大的胰腺参考结果，不直接纳入 git。
复现已发布的胰腺 checkpoint 时，还需要传入
`gene_order_path="data/pancreas/reference_checkpoints/pancreas_genes.txt"`；
模型权重和 attention tensor 按 gene index 对齐，同一批基因但顺序不同并不等价。

模块化用法
----------
```python
from gravity import (
    preprocess_counts,
    resolve_gene_order,
    CellStageConfig, train_cell_stage,
    GeneStageConfig, train_gene_stage,
)
from gravity.tools.future import estimate_future_positions
from gravity.plotting.velocity import plot_velocity_cell, plot_velocity_gene

RAW_COUNTS = "data/PancreaticEndocrinogenesis_cell_type_u_s.csv"
WORKDIR = "gravity_outputs_pancreas"
PRIOR_NET = "prior_data/nichenet_mouse.zip"
GENE_ORDER = "data/pancreas/reference_checkpoints/pancreas_genes.txt"
genes = resolve_gene_order(None, GENE_ORDER)

middle_csv = preprocess_counts(
    RAW_COUNTS,
    f"{WORKDIR}/combine.csv",
    gene_order=genes,
)

cell_cfg = CellStageConfig(
    raw_counts=RAW_COUNTS,
    middle_csv=str(middle_csv),
    prior_network=PRIOR_NET,
    output_dir=WORKDIR,
    gene_subset=genes,
    gene_order_path=GENE_ORDER,
    accelerator="gpu",
    devices=1,
    batch_size=16,
    learning_rate=1e-6,
)
stage1 = train_cell_stage(cell_cfg)

estimate_future_positions(stage1["stage1_csv"], f"{WORKDIR}/future_positions.npy")

gene_cfg = GeneStageConfig(
    raw_counts=RAW_COUNTS,
    middle_csv=str(middle_csv),
    stage1_checkpoint=str(stage1["checkpoint"]),
    future_positions=f"{WORKDIR}/future_positions.npy",
    prior_network=PRIOR_NET,
    output_dir=WORKDIR,
    gene_subset=genes,
    gene_order_path=GENE_ORDER,
    accelerator="gpu",
    devices=1,
    batch_size=16,
    epochs=4,
    learning_rate=1e-4,
)
stage2 = train_gene_stage(gene_cfg)

plot_velocity_cell(str(stage2["stage2_csv"]), output_path=f"{WORKDIR}/cell_velocity.png")
plot_velocity_gene(str(stage2["stage2_csv"]), gene="INS2", output_path=f"{WORKDIR}/ins2_velocity_expression.png")
```

关键配置
--------
- `PipelineConfig`: `gene_subset`、`gene_order_path`、`stage*_epochs`、`stage*_lr`、`val_fraction_stage*`（默认为 0，无验证集）、`future_tau`、`accelerator/devices/strategy`、`make_plot/plot_genes`；
- `CellStageConfig`: `attention_output`、`attention_topk`；
- `GeneStageConfig`: `future_positions`、`stage1_checkpoint`。

输入格式
--------
GRAVITY 参考 cellDancer 的长表计数存储形式。CSV 至少包含：`cellID`、
`gene_name`、`unsplice`、`splice`、`embedding1`、`embedding2`；可选
`clusters` 用于上色。先验网络应是 zipped CSV edge list，包含 `from` 和
`to` gene 列；仓库已在 `prior_data/` 下提供 mouse 和 human 的
NicheNet-derived 网络。
大型原始计数表不直接存放在仓库中；pancreatic endocrinogenesis smoke test 默认使用的路径见 `data/README.md`。

常见问题
--------
- 显存不足：降低 `batch_size` 或缩小 `gene_subset`；
- 无 GPU：Lightning 自动回退至 CPU；
- 缺少可选依赖：绘图/采样可能自动禁用；
- 日志：使用 `gravity.utils.log_verbose` 等工具。

贡献与许可
----------
欢迎提交 Issue/PR（请附复现步骤与命令序列）。本软件包遵循 MIT 许可。
