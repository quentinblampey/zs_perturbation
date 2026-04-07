from pathlib import Path

import zs_perturbation
from zs_perturbation import CONTROL, DISEASES


def main(disease: str) -> None:
    adata = zs_perturbation.load_dataset(disease)

    adata_control = adata[adata.obs["disease"] == CONTROL].copy()
    adata_disease = adata[adata.obs["disease"] != CONTROL].copy()

    healthy_centroid = zs_perturbation.encode(adata_control).mean(dim=0)

    scores = zs_perturbation.compute_encoder_score(adata_disease, healthy_centroid)
    adata.var["encoder_scores"] = scores

    df_res = zs_perturbation.extract_scores(adata, disease, "encoder_scores")

    output = Path(".") / "results" / "v1" / f"{disease}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output)


if __name__ == "__main__":
    zs_perturbation.download_dataset()

    for i, disease in enumerate(DISEASES):
        print(f"Processing {disease} ({i + 1}/{len(DISEASES)})")
        main(disease)
