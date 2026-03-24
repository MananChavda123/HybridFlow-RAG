from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased")
print(model)
