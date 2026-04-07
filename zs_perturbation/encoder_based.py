import numpy as np
import torch
from anndata import AnnData
from tqdm.auto import tqdm

from .eva import model, tokenizer


def encode(adata: AnnData, device: str = "cpu", batch_size: int = 25) -> torch.Tensor:
    outputs = []

    token_ids = tokenizer.convert_tokens_to_ids(adata.var_names)
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    for i in tqdm(range(0, adata.n_obs, batch_size), desc="Encoding"):
        batch_adata = adata[i : i + batch_size]

        batch_genes = token_ids_tensor.unsqueeze(0).expand(batch_adata.n_obs, -1)
        batch_values = torch.from_numpy(batch_adata.X.astype(np.float32)).to(device)

        with torch.inference_mode():
            outputs.append(model.encode(gene_ids=batch_genes, expression_values=batch_values)["cls_embedding"])

    return torch.cat(outputs, dim=0)


def compute_encoder_score(
    adata: AnnData, healthy_centroid: torch.Tensor, device: str = "cpu", batch_size: int = 25
) -> np.ndarray:
    grads = []

    token_ids = tokenizer.convert_tokens_to_ids(adata.var_names)
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    for i in tqdm(range(0, adata.n_obs, batch_size), desc="Gradient Computation"):
        batch_adata = adata[i : i + batch_size]

        batch_genes = token_ids_tensor.unsqueeze(0).expand(batch_adata.n_obs, -1)
        batch_values = torch.from_numpy(batch_adata.X.astype(np.float32)).to(device)
        batch_values.requires_grad = True

        cls_embedding = model.encode(gene_ids=batch_genes, expression_values=batch_values)["cls_embedding"]

        loss = ((cls_embedding - healthy_centroid) ** 2).sum(axis=1).mean()
        loss.backward()

        grads.append(batch_values.grad.numpy(force=True))

    grads = np.concatenate(grads, axis=0)

    mean_grads = grads.mean(0)

    return mean_grads / np.linalg.norm(mean_grads)
