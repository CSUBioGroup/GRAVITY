# Run the GRAVITY Pipeline

Once your CSV is ready, run the two-stage pipeline via `PipelineConfig` and `run_pipeline`.

```python
from gravity import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    raw_counts="data/pancreas_long.csv",
    workdir="gravity_outputs",
    prior_network="prior_data/network_mouse.zip",
    accelerator="gpu",
    devices=[0],
    strategy="ddp",
    make_plot=True,
    plot_genes=["GCG", "INS1"],
    stage1_epochs=6,
    stage2_epochs=6,
)
outputs = run_pipeline(cfg)
print(outputs)
```

Key tips:

- Use unique `workdir` names per experiment to avoid overwriting checkpoints.
- Set `devices` and `strategy` to match your cluster (e.g., `devices=[0,1]`, `strategy="ddp"`).
- Reduce `batch_size` or provide `gene_subset` when GPU memory is limited.

The resulting dictionary contains paths to `combine.csv`, stage checkpoints, `future_positions.npy`, and attention exports.
