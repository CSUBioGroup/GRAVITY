# Prior Network Data

This directory contains species-specific prior network archives used by
GRAVITY:

```text
nichenet_mouse.zip
nichenet_human.zip
```

Each archive contains one CSV edge list. The external archive name is the public
path used by GRAVITY; the CSV name inside the zip is kept compatible with the
original source layout and is read automatically by pandas.

The archives follow the prior-network processing described by CEFCON. They
start from NicheNet's integrated gene interaction network, remove cell-cell
ligand-receptor interactions, use the unweighted integrated network, and
represent undirected edges as bidirectional directed edges. The human archive
keeps human gene symbols, and the mouse archive uses one-to-one ENSEMBL
ortholog mapping with ambiguous genes removed. The mouse archive is the default
for the pancreas demo; use the human archive for human datasets:

```python
from gravity import PipelineConfig

cfg = PipelineConfig(
    raw_counts="data/your_human_counts.csv",
    prior_network="prior_data/nichenet_human.zip",
)
```

Background links:

- NicheNet: https://www.nature.com/articles/s41592-019-0667-5
- CEFCON: https://www.nature.com/articles/s41467-023-44103-3
- cellDancer: https://www.nature.com/articles/s41587-023-01728-5
