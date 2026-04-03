from pathlib import Path

import anndata
import pandas as pd
import scanpy as sc
from anndata import AnnData
from huggingface_hub import snapshot_download

from . import BENCHMARK_FILE, DATASET_NAME, TO_GENE_SYMBOL


def download_dataset() -> None:
    """
    Downloads the dataset from Hugging Face Hub inside the current directory.
    """
    if Path(DATASET_NAME).exists():
        print(f"Dataset '{DATASET_NAME}' already exists. Skipping download.")
        return

    snapshot_download(repo_id=f"ScientaLab/{DATASET_NAME}", repo_type="dataset", local_dir=DATASET_NAME)


def load_dataset(disease_abbrev: str) -> AnnData:
    adata = anndata.read_h5ad(Path(DATASET_NAME) / disease_abbrev / "dataset.h5ad")

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")

    selection = adata.var["highly_variable"] | adata.var["gene_symbols"].isin(_genes_of_interest())
    adata = adata[:, selection].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def _genes_of_interest() -> list[str]:
    df_benchmark = pd.read_csv(BENCHMARK_FILE, index_col=0)

    genes = df_benchmark.target_genes.str.split(";").explode().unique()
    genes = [TO_GENE_SYMBOL.get(gene, gene) for gene in genes]

    return genes
