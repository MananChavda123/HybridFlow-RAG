from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Example usage
text = "Hello! This is a test."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
