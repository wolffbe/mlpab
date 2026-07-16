import inspect
import hsfs.embedding as embedding

print([n for n in dir(embedding) if not n.startswith('_')])
print(inspect.signature(embedding.EmbeddingIndex.__init__))
print(inspect.signature(embedding.EmbeddingIndex.add_embedding))
sft = getattr(embedding, 'SimilarityFunctionType', None)
print(sft, [n for n in dir(sft) if not n.startswith('_')] if sft else None)
