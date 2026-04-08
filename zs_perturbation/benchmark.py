import pandas as pd
from anndata import AnnData

from . import BENCHMARK_FILE, TO_GENE_SYMBOL

DF_BENCH = pd.read_csv(BENCHMARK_FILE, index_col=0)


def genes_of_interest(disease_abbrev: str) -> list[str]:
    df_bench_disease = DF_BENCH[DF_BENCH["disease_abbrev"] == disease_abbrev]

    genes = df_bench_disease.target_genes.str.split(";").explode().unique()
    genes = [TO_GENE_SYMBOL.get(gene, gene) for gene in genes]

    return genes


def extract_scores(adata: AnnData, disease_abbrev: str, var_key: str) -> pd.DataFrame:
    """Compute the target efficacy score and ground truth for a given method.

    Args:
        adata: The AnnData object containing the genes scores.
        disease_abbrev: The disease abbreviation to filter the benchmark dataframe.
        var_key: The key in adata.var where the gene scores are stored.

    Returns:
        A DataFrame of shape (n_samples, 2) with columns 'y_true' and 'y_score' where 'y_true' is a binary array
        indicating the expected efficacy of the drug, and 'y_score' is an array
        containing the computed scores for each drug.
    """
    to_entrez_id = pd.Series(adata.var_names, index=adata.var["gene_symbols"])

    def _score(genes: list[str]) -> float:
        entrez_ids = to_entrez_id[[TO_GENE_SYMBOL.get(gene, gene) for gene in genes]].values
        return adata.var.loc[entrez_ids, var_key].mean()

    df_disease = DF_BENCH[DF_BENCH["disease_abbrev"] == disease_abbrev]

    y_score = df_disease.target_genes.str.split(";").apply(_score).values
    y_true = df_disease["expected_efficacy"].astype(int).values

    return pd.DataFrame({"y_true": y_true, "y_score": y_score}, index=df_disease.index)
