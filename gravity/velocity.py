"""Visualization and velocity helpers packaged for GRAVITY (legacy behavior)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .utils import log_verbose

try:  # Optional SciPy dependency for advanced sampling
    from scipy.stats import gaussian_kde, rv_discrete
except ImportError:  # pragma: no cover
    gaussian_kde = None
    rv_discrete = None

__all__ = [
    "extract_from_df",
    "retain_top_1_percent",
    "sampling_neighbors",
    "sampling_embedding",
    "downsampling_embedding",
    "corr_coeff",
    "data_reshape",
    "compute_cell_velocity_",
]


# ----------------------------
# Basic utilities
# ----------------------------
def extract_from_df(load_cellDancer: pd.DataFrame,
                    attr_list: Sequence[str],
                    gene_name: Optional[str] = None) -> np.ndarray:
    if gene_name is None:
        gene_name = load_cellDancer.gene_name.iloc[0]
    one_gene_idx = load_cellDancer.gene_name == gene_name
    data = load_cellDancer[one_gene_idx][attr_list].dropna()
    return data.to_numpy()


def retain_top_1_percent(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros_like(matrix)
    if matrix.size == 0:
        return result
    thresholds = np.percentile(matrix, 99, axis=1)
    for idx, threshold in enumerate(thresholds):
        result[idx] = np.where(matrix[idx] >= threshold, matrix[idx], 0)
    return result


# ----------------------------
# Sampling (legacy strategy)
# ----------------------------
def _gaussian_kernel(X: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    return np.exp(-(X - mu) ** 2 / (2.0 * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)


def sampling_neighbors(points: np.ndarray,
                       step: Tuple[int, int] = (30, 30),
                       percentile: float = 25.0,
                       seed: int = 10) -> np.ndarray:
    """Legacy sampling:
      1) Regular grid + N(0,0.15) noise (fixed seed)
      2) For each grid anchor, pick nearest original point → unique indices
      3) Second KNN on candidates; select by Gaussian-kernel density percentile
    """
    if points.shape[0] == 0:
        return np.array([], dtype=int)

    # 1) Grid + noise
    grids = []
    for dim in range(points.shape[1]):
        minimum, maximum = points[:, dim].min(), points[:, dim].max()
        pad = 0.025 * abs(maximum - minimum)
        grid = np.linspace(minimum - pad, maximum + pad, step[dim])
        grids.append(grid)

    mesh = np.meshgrid(*grids)
    grid_coords = np.vstack([axis.flat for axis in mesh]).T
    if seed is not None:
        np.random.seed(int(seed))
    grid_coords = grid_coords + np.random.normal(loc=0.0, scale=0.15, size=grid_coords.shape)

    # 2) Grid anchors → candidate indices
    neighbors = min(points.shape[0] - 1, 20)
    if neighbors <= 0:
        return np.arange(points.shape[0], dtype=int)

    nn = NearestNeighbors()
    nn.fit(points[:, :2])
    dist1, ixs1 = nn.kneighbors(grid_coords, neighbors)  # distances, indices
    ix_choice = np.unique(ixs1[:, 0])
    if ix_choice.size == 0:
        return ix_choice  # Preserve legacy behavior; no fallback

    # 3) Second KNN distances → Gaussian kernel density → percentile threshold
    dist2, _ = nn.kneighbors(points[ix_choice, :2], neighbors)  # distances
    density = _gaussian_kernel(dist2, mu=0.0, sigma=0.5).sum(axis=1)
    thr = np.percentile(density, percentile)
    mask = density > thr  # Legacy uses strict greater-than
    return ix_choice[mask]


def _check_scipy(feature: str) -> None:
    if gaussian_kde is None or rv_discrete is None:
        raise ImportError(
            f"SciPy is required for '{feature}' sampling. Install scipy or select a different mode."
        )


def sampling_inverse(points: np.ndarray, target_amount: int = 500) -> np.ndarray:
    _check_scipy("inverse")
    kde = gaussian_kde(points.T)
    probabilities = kde(points.T)
    weights = (1.0 / probabilities)
    weights /= weights.sum()
    idx = np.arange(points.shape[0])
    sampler = rv_discrete(values=(idx, weights))
    return sampler.rvs(size=target_amount)


def sampling_circle(points: np.ndarray, target_amount: int = 500) -> np.ndarray:
    _check_scipy("circle")
    kde = gaussian_kde(points.T)
    probabilities = kde(points.T)
    adjusted = np.square(1 - (probabilities / probabilities.max()) ** 2) + 1e-4
    adjusted /= adjusted.sum()
    idx = np.arange(points.shape[0])
    sampler = rv_discrete(values=(idx, adjusted))
    return sampler.rvs(size=target_amount)


def sampling_random(points: np.ndarray, target_amount: int = 500) -> np.ndarray:
    return np.random.choice(points.shape[0], size=target_amount, replace=False)


def sampling_embedding(detail: pd.DataFrame,
                       para: str,
                       target_amount: int = 500,
                       step: Tuple[int, int] = (30, 30)) -> np.ndarray:
    """When ``para='neighbors'``, call :func:`sampling_neighbors` (percentile=25)."""
    values = detail[["embedding1", "embedding2"]].to_numpy()
    if para == 'neighbors':
        return sampling_neighbors(values, step)
    if para == 'inverse':
        return sampling_inverse(values, target_amount)
    if para == 'circle':
        return sampling_circle(values, target_amount)
    if para == 'random':
        return sampling_random(values, target_amount)
    raise ValueError("para is expected to be one of {'neighbors','inverse','circle','random'}")


# ----------------------------
# Downsampling for velocity
# ----------------------------
def downsampling_embedding(data_df: pd.DataFrame,
                           para: str,
                           target_amount: int,
                           step: Optional[Tuple[int, int]],
                           n_neighbors: int,
                           expression_scale: Optional[str] = None,
                           projection_neighbor_choice: str = 'embedding',
                           pca_n_components: Optional[int] = None,
                           umap_n: Optional[int] = None,
                           umap_n_components: Optional[int] = None):
    """
    Returns: ``embedding_downsampling, idx_downsampling, embedding_knn``.
    If ``projection_neighbor_choice='embedding'``, the neighbor graph is built
    on the downsampled embedding (legacy behavior).
    """
    gene = data_df['gene_name'].drop_duplicates().iloc[0]
    embedding = data_df.loc[data_df['gene_name'] == gene][['embedding1', 'embedding2']]

    if step is not None:
        idx_downsampling = sampling_embedding(embedding, para=para, target_amount=target_amount, step=step)
    else:
        idx_downsampling = np.arange(embedding.shape[0])

    # Optional expression scaling on splice/unsplice
    if expression_scale is not None:
        scaled = data_df.copy()
        if expression_scale == 'log':
            scaled['splice'] = np.log(scaled['splice'] + 1e-6)
            scaled['unsplice'] = np.log(scaled['unsplice'] + 1e-6)
        elif expression_scale == '2power':
            scaled['splice'] = np.power(2.0, scaled['splice'])
            scaled['unsplice'] = np.power(2.0, scaled['unsplice'])
        elif expression_scale == 'power10':
            scaled['splice'] = np.power(scaled['splice'], 10)
            scaled['unsplice'] = np.power(scaled['unsplice'], 10)
        else:
            log_verbose(f"Unknown expression_scale '{expression_scale}', using raw values.", level=1)
        data_df = scaled

    if projection_neighbor_choice == 'gene':
        cell_ids = data_df.loc[data_df['gene_name'] == gene]['cellID']
        pivot = data_df.pivot(index='cellID', columns='gene_name', values='splice').reindex(cell_ids)
        embedding_downsampling = pivot.iloc[idx_downsampling]
    elif projection_neighbor_choice == 'pca':  # pragma: no cover
        from sklearn.decomposition import PCA
        cell_ids = data_df.loc[data_df['gene_name'] == gene]['cellID']
        pivot = data_df.pivot(index='cellID', columns='gene_name', values='splice').reindex(cell_ids)
        embedding_pre = pivot.iloc[idx_downsampling]
        if pca_n_components is None:
            raise ValueError("pca_n_components must be provided when using projection_neighbor_choice='pca'")
        pca = PCA(n_components=pca_n_components)
        embedding_downsampling = pca.fit_transform(embedding_pre)
    elif projection_neighbor_choice == 'embedding':
        embedding_downsampling = embedding.iloc[idx_downsampling][['embedding1', 'embedding2']]
    elif projection_neighbor_choice == 'umap':  # pragma: no cover
        import umap
        cell_ids = data_df.loc[data_df['gene_name'] == gene]['cellID']
        pivot = data_df.pivot(index='cellID', columns='gene_name', values='splice').reindex(cell_ids)
        embedding_pre = pivot.iloc[idx_downsampling]
        reducer = umap.UMAP(
            n_neighbors=umap_n or 15,
            min_dist=0.1,
            n_components=umap_n_components or 2,
            metric='euclidean',
        )
        embedding_downsampling = reducer.fit_transform(embedding_pre)
    else:
        raise ValueError(f"Unsupported projection_neighbor_choice '{projection_neighbor_choice}'")

    n_neighbors = min(max(1, embedding_downsampling.shape[0] // 4), n_neighbors)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embedding_downsampling)
    embedding_knn = nn.kneighbors_graph(mode="connectivity")
    return embedding_downsampling, idx_downsampling, embedding_knn


# ----------------------------
# Velocity math
# ----------------------------
def corr_coeff(ematrix: np.ndarray, vmatrix: np.ndarray, index: int) -> np.ndarray:
    ematrix = ematrix.T
    vmatrix = vmatrix.T
    ematrix = ematrix - ematrix[index, :]
    vmatrix = vmatrix[index, :][None, :]
    ematrix_m = ematrix - ematrix.mean(axis=1)[:, None]
    vmatrix_m = vmatrix - vmatrix.mean(axis=1)[:, None]
    ematrix_ss = (ematrix_m ** 2).sum(axis=1)
    vmatrix_ss = (vmatrix_m ** 2).sum(axis=1)
    numerator = np.dot(ematrix_m, vmatrix_m.T)
    denominator = np.sqrt(np.dot(ematrix_ss[:, None], vmatrix_ss[None]))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0).T


def data_reshape(cellDancer_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - np_splice:   (ngenes, ncells)
      - np_d_matrix: (ngenes, ncells) with sqrt(abs(.)) * sign(.) transform
    Also inserts an 'index' column (per gene, 0..ncell-1) into the dataframe.
    """
    psc = 1.0
    gene_names = cellDancer_df['gene_name'].drop_duplicates().to_list()
    cell_number = cellDancer_df[cellDancer_df['gene_name'] == gene_names[0]].shape[0]
    cellDancer_df['index'] = np.tile(range(cell_number), len(gene_names))

    splice_reshape = cellDancer_df.pivot(index='gene_name', values='splice', columns='index')
    splice_predict_reshape = cellDancer_df.pivot(index='gene_name', values='splice_predict', columns='index')
    d_matrix = splice_predict_reshape - splice_reshape
    np_splice = np.asarray(splice_reshape)
    np_d_matrix = np.asarray(d_matrix)
    np_d_matrix2 = np.sqrt(np.abs(np_d_matrix) + psc) * np.sign(np_d_matrix)
    return np_splice, np_d_matrix2


def compute_cell_velocity_(cellDancer_df: pd.DataFrame,
                           gene_list: Optional[Sequence[str]] = None,
                           speed_up: Optional[Tuple[int, int]] = (60, 60),
                           expression_scale: Optional[str] = None,
                           projection_neighbor_size: int = 200,
                           projection_neighbor_choice: str = 'embedding') -> pd.DataFrame:
    """Fully aligned with legacy behavior:
      - Return a single DataFrame (not a tuple)
      - Only indices in ``sampling_ixs`` (per-gene 0..ncell-1) have velocity
      - Copy velocity across all genes for rows with ``cellIndex ∈ sampling_ixs``
    """

    def velocity_correlation(cell_matrix: np.ndarray, velocity_matrix: np.ndarray) -> np.ndarray:
        corr = np.zeros((cell_matrix.shape[1], velocity_matrix.shape[1]))
        for idx in range(cell_matrix.shape[1]):
            corr[idx, :] = corr_coeff(cell_matrix, velocity_matrix, idx)[0, :]
        np.fill_diagonal(corr, 0)
        return corr

    def velocity_projection(cell_matrix: np.ndarray,
                            velocity_matrix: np.ndarray,
                            embedding: np.ndarray,
                            knn_embedding) -> np.ndarray:
        sigma_corr = 0.05
        cell_matrix = np.nan_to_num(cell_matrix)
        velocity_matrix = np.nan_to_num(velocity_matrix)
        corrcoef = velocity_correlation(cell_matrix, velocity_matrix)
        probability_matrix = np.exp(corrcoef / sigma_corr) * knn_embedding.A
        probability_matrix /= probability_matrix.sum(1)[:, None]
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
            np.fill_diagonal(unitary_vectors[0, ...], 0)
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        velocity_embedding = (probability_matrix * unitary_vectors).sum(2)
        velocity_embedding -= (knn_embedding.A * unitary_vectors).sum(2) / knn_embedding.sum(1).A.T
        return velocity_embedding.T

    # Filter out invalid predictions
    filtered = cellDancer_df.dropna(subset=['alpha', 'beta']).reset_index(drop=True)
    if gene_list is None:
        gene_list = filtered.gene_name.drop_duplicates()

    cellDancer_df_input = filtered[filtered.gene_name.isin(gene_list)].reset_index(drop=True)
    np_splice_all, np_d_matrix_all = data_reshape(cellDancer_df_input)
    n_genes, _ = np_splice_all.shape

    # Prepare data for graph/neighbors (legacy)
    data_df = cellDancer_df_input.loc[:, ['gene_name', 'unsplice', 'splice', 'cellID', 'embedding1', 'embedding2']]
    embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(
        data_df,
        para='neighbors',
        target_amount=0,
        step=speed_up,
        n_neighbors=projection_neighbor_size,
        projection_neighbor_choice=projection_neighbor_choice,
        expression_scale=expression_scale,
        pca_n_components=None,
        umap_n=None,
        umap_n_components=None,
    )

    # Note: use the first gene's embedding (legacy)
    first_gene = list(gene_list)[0]
    embedding = cellDancer_df_input[cellDancer_df_input.gene_name == first_gene][['embedding1', 'embedding2']].to_numpy()

    velocity_embedding = velocity_projection(
        np_splice_all[:, sampling_ixs],
        np_d_matrix_all[:, sampling_ixs],
        embedding[sampling_ixs, :],
        knn_embedding,
    )

    # Overwrite existing velocity columns if present
    if {'velocity1', 'velocity2'}.issubset(cellDancer_df_input.columns):
        log_verbose("Caution! Overwriting the 'velocity' columns.", level=1)
        cellDancer_df_input = cellDancer_df_input.drop(columns=['velocity1', 'velocity2'])

    # Key: propagate strictly by ``cellIndex ∈ sampling_ixs`` (legacy)
    sampling_ixs_all_genes = cellDancer_df_input[cellDancer_df_input['cellIndex'].isin(sampling_ixs)].index
    cellDancer_df_input.loc[sampling_ixs_all_genes, 'velocity1'] = np.tile(velocity_embedding[:, 0], n_genes)
    cellDancer_df_input.loc[sampling_ixs_all_genes, 'velocity2'] = np.tile(velocity_embedding[:, 1], n_genes)

    log_verbose(f"After downsampling, there are {len(sampling_ixs)} cells.", level=1)
    return cellDancer_df_input, velocity_embedding
