GRAVITY：Dynamic gene regulatory network-enhanced RNA velocity modeling for trajectory inference and biological discovery
=========================================================================================================================

本子包提供重构后的 GRAVITY 模块化实现（全称如标题所示），强调研究友好的接口与清晰的模块边界，包含：预处理、两阶段训练、未来位置估计、可视化，以及 TF 重要性分析。

特性
----
- 单一配置对象的一键端到端流程；
- 两阶段训练（细胞层→基因层），支持多 GPU；
- 结合调控先验的未来位置估计与可视化；
- 导出 TF 注意力得分矩阵，便于下游分析。

安装
----
建议使用 Python 3.9+ 并创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如需 GPU，请先安装对应的 `torch` CUDA 版本，然后执行 `pip install -e .`。

快速开始（端到端）
-----------------
```python
from gravity import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    raw_counts="data/pancreas_long.csv",
    workdir="gravity_outputs",
    prior_network="prior_data/network_mouse.zip",
    accelerator="gpu",
    devices=[0, 1],
    strategy="ddp",
    make_plot=True,
    plot_genes=["GCG", "INS1"],
)
outputs = run_pipeline(cfg)
print(outputs)
```

从 h5ad 转换为 CSV
-------------------
若数据源是 AnnData，可先使用 `export_intermediate_from_h5ad` 转成 GRAVITY 需要的长表 CSV：

```python
from gravity import export_intermediate_from_h5ad

export_intermediate_from_h5ad(
    input_h5ad="data/postprocessed.h5ad",
    output_csv="data/hair.csv",
    n_top_genes=1000,
    embed_key="X_umap",
    celltype_key="celltype",
)
```
该流程与 `gravity/smoke_test_hair.py` 中示例一致，会同时保留下游所需的嵌入与聚类标签。

完成后，`workdir` 通常包含：`combine.csv`、`stage1.csv`/`stage1.ckpt`、`future_positions.npy`、`stage2.csv`/`stage2.ckpt`、`attentions/`、`velocity_plots/*.png`（若开启绘图）。

模块化用法
----------
```python
from gravity import (
    preprocess_counts,
    CellStageConfig, train_cell_stage,
    GeneStageConfig, train_gene_stage,
)
from gravity.tools.future import estimate_future_positions
from gravity.plotting.velocity import plot_velocity_cell, plot_velocity_gene
```

关键配置
--------
- `PipelineConfig`: `gene_subset`、`stage*_epochs`、`val_fraction_stage*`（默认为 0，无验证集）、`future_tau`、`accelerator/devices/strategy`、`make_plot/plot_genes`；
- `CellStageConfig`: `attention_output`、`attention_topk`；
- `GeneStageConfig`: `future_positions`、`stage1_checkpoint`。

输入格式
--------
长表 CSV 至少包含：`cellID`、`gene_name`、`unsplice`、`splice`、`embedding1`、`embedding2`；可选 `clusters` 用于上色。先验压缩包 `prior_data/network_mouse.zip` 与原始格式保持一致。

常见问题
--------
- 显存不足：降低 `batch_size` 或缩小 `gene_subset`；
- 无 GPU：Lightning 自动回退至 CPU；
- 缺少可选依赖：绘图/采样可能自动禁用；
- 日志：使用 `gravity.utils.log_verbose` 等工具。

引用
----
如在研究中使用本包，请引用 GRAVITY 相关论文/项目（完整标题：GRAVITY: Dynamic gene regulatory network-enhanced RNA velocity modeling for trajectory inference and biological discovery）；RNA velocity 工具背景可参考 scVelo 文献与其项目主页。

贡献与许可
----------
欢迎提交 Issue/PR（请附复现步骤与命令序列）。本子包遵循 MIT 许可。
