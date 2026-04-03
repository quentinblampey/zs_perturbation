from huggingface_hub import snapshot_download

from . import DATASET_NAME


def download_dataset() -> None:
    """
    Downloads the dataset from Hugging Face Hub inside the current directory.
    """
    snapshot_download(repo_id=f"ScientaLab/{DATASET_NAME}", repo_type="dataset", local_dir=DATASET_NAME)
