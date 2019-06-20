
print("finally, let's do some ranking...")
entity_count = 8
scores, _, _ = comparator(
    comparator.prepare(torch.tensor(src_embedding.reshape([1,1,10]))).expand(1, entity_count, 10),
    comparator.prepare(torch.tensor(dest_embeddings.reshape([1,8,10]))),
    torch.empty(1, 0, 10),  # Left-hand side negatives, not needed
    torch.empty(1, 0, 10),  # Right-hand side negatives, not needed
)
permutation = scores.flatten().argsort(descending=True)

top_entities = [dictionary["entities"]["user_id"][index] for index in permutation]
print(top_entities)
### SNIPPET 3 ###