from fastembed import TextEmbedding

# List all available models
print("✅ Available embedding models in FastEmbed:\n")
for model in TextEmbedding.list_supported_models():
    print("-", model)
