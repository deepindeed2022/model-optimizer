from datasets import load_dataset

# data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# # \t is the tab character in Python
# drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
# result = unmasker("This programmer is wearing [MASK].")
print("This man works as a ", [r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print("This woman works as a ", [r["token_str"] for r in result])