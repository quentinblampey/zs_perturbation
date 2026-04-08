import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from tqdm.auto import tqdm

from .eva import model, tokenizer

z_holder: dict[str, torch.Tensor] = {}


def register_hook():
    def hook_fn(module, _, output):
        z_holder["z"] = output
        output.retain_grad()

    model.layers[-2].register_forward_hook(hook_fn)


def store_z_intermediate(adata: AnnData, device: str = "cpu", batch_size: int = 10) -> None:
    adata.obsm["z_intermediate"] = np.zeros((adata.n_obs, 256))

    token_ids = tokenizer.convert_tokens_to_ids(adata.var_names)
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    for i in tqdm(range(0, adata.n_obs, batch_size), desc="Encoding"):
        batch_adata = adata[i : i + batch_size]

        batch_genes = token_ids_tensor.unsqueeze(0).expand(batch_adata.n_obs, -1)
        batch_values = torch.from_numpy(batch_adata.X.astype(np.float32)).to(device)

        model.encode(gene_ids=batch_genes, expression_values=batch_values)

        adata.obsm["z_intermediate"][i : i + batch_size] = z_holder["z"][:, 0].numpy(force=True)


def compute_healthy_score(
    adata: AnnData,
    healthy_centroid: np.ndarray,
    genes: list[str],
    device: str = "cpu",
    batch_size: int = 10,
) -> None:
    adata.var["mean_healthy_score"] = 0.0

    adata.obsm["z_intermediate"] = np.zeros((adata.n_obs, 256))
    adata.obsm["grad"] = np.zeros((adata.n_obs, 256))

    token_ids = tokenizer.convert_tokens_to_ids(adata.var_names)
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    to_entrez_id = pd.Series(adata.var_names, index=adata.var["gene_symbols"])

    for gene_index, gene in enumerate(genes):
        print(f"Processing gene {gene} ({gene_index + 1}/{len(genes)})")
        entrez_id = to_entrez_id[gene]

        for i in tqdm(range(0, adata.n_obs, batch_size), desc="Processing batches"):
            batch_adata = adata[i : i + batch_size]

            batch_genes = token_ids_tensor.unsqueeze(0).expand(batch_adata.n_obs, -1)
            batch_values = torch.from_numpy(batch_adata.X.astype(np.float32)).to(device)

            output = model.encode(gene_ids=batch_genes, expression_values=batch_values)

            predicted_expression = model.decode(output.gene_embeddings)

            loss = -predicted_expression[:, adata.var_names.get_loc(entrez_id)].mean()
            loss.backward()

            adata.obsm["z_intermediate"][i : i + batch_size] = z_holder["z"][:, 0].numpy(force=True)

            grad = z_holder["z"].grad[:, 0].numpy(force=True)
            adata.obsm["grad"][i : i + batch_size] = grad / np.linalg.norm(grad, axis=1, keepdims=True)
            z_holder["z"].grad.zero_()

        healthy_direction = healthy_centroid - adata.obsm["z_intermediate"].mean(0)
        healthy_direction /= np.linalg.norm(healthy_direction)

        adata.obs["healthy_score"] = adata.obsm["grad"] @ healthy_direction

        adata.var.loc[entrez_id, "mean_healthy_score"] = adata.obs["healthy_score"].mean()
