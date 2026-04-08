import argparse
from pathlib import Path

import zs_perturbation
from zs_perturbation import CONTROL, DISEASES, genes_of_interest


def save_scores(disease_abbrev: str, method: str) -> None:
    adata = zs_perturbation.load_dataset(disease_abbrev)

    adata_control = adata[adata.obs["disease"] == CONTROL].copy()
    adata_disease = adata[adata.obs["disease"] != CONTROL].copy()

    if method == "encoder":
        healthy_centroid = zs_perturbation.encode(adata_control).mean(dim=0)

        scores = zs_perturbation.compute_encoder_score(adata_disease, healthy_centroid)
        adata_disease.var["encoder_scores"] = scores

        df_res = zs_perturbation.extract_scores(adata_disease, disease_abbrev, "encoder_scores")

    elif method == "decoder":
        zs_perturbation.store_z_intermediate(adata_control)
        healthy_centroid = adata_control.obsm["z_intermediate"].mean(axis=0)

        zs_perturbation.compute_healthy_score(adata_disease, healthy_centroid, genes_of_interest(disease_abbrev))
        df_res = zs_perturbation.extract_scores(adata_disease, disease_abbrev, "mean_healthy_score")

    else:
        raise ValueError(f"Unknown method: {method}")

    output = Path(".") / "results" / method / f"{disease_abbrev}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output)


def initialize_run(method: str) -> None:
    zs_perturbation.download_dataset()

    if method == "decoder":
        zs_perturbation.register_hook()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the benchmark for a given disease.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["encoder", "decoder"],
        default="encoder",
        help="The method to use for computing the scores.",
    )
    args = parser.parse_args()

    initialize_run(args.method)

    for i, disease_abbrev in enumerate(DISEASES):
        print(f"Processing {disease_abbrev} ({i + 1}/{len(DISEASES)})")
        save_scores(disease_abbrev, args.method)
