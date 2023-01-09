import transformers

def get_image_token_embedding(word_embedding):
    config = transformers.LayoutLMv3Config()
    config.hidden_size = word_embedding.embedding_dim
    config.padding_idx = word_embedding.padding_idx
    config.shape_size = 512
    config.coordinate_size = 768        
    embedding = transformers.models.layoutlmv3.modeling_layoutlmv3.LayoutLMv3TextEmbeddings(config)
    embedding.word_embeddings = word_embedding
    return embedding