from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ScientaLab/eva-rna", trust_remote_code=True)
