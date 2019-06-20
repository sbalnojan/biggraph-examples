### SNIPPET 1 ###

print("Now let's do some simple things within torch:")

from torchbiggraph.model import DotComparator
src_entity_offset = dictionary["entities"]["user_id"].index("0")  # France
dest_1_entity_offset = dictionary["entities"]["user_id"].index("7")  # Paris
dest_2_entity_offset = dictionary["entities"]["user_id"].index("1")  # Paris
rel_type_index = dictionary["relations"].index("follow") # note we only have one...

with h5py.File("model/example_2/embeddings_user_id_0.v10.h5", "r") as hf:
    src_embedding = hf["embeddings"][src_entity_offset, :]
    dest_1_embedding = hf["embeddings"][dest_1_entity_offset, :]
    dest_2_embedding = hf["embeddings"][dest_2_entity_offset, :]
    dest_embeddings = hf["embeddings"][...]


import torch
comparator = DotComparator()

scores_1, _, _ = comparator(
    comparator.prepare(torch.tensor(src_embedding.reshape([1,1,10]))),
    comparator.prepare(torch.tensor(dest_1_embedding.reshape([1,1,10]))),
    torch.empty(1, 0, 10),  # Left-hand side negatives, not needed
    torch.empty(1, 0, 10),  # Right-hand side negatives, not needed
)

scores_2, _, _ = comparator(
    comparator.prepare(torch.tensor(src_embedding.reshape([1,1,10]))),
    comparator.prepare(torch.tensor(dest_2_embedding.reshape([1,1,10]))),
    torch.empty(1, 0, 10),  # Left-hand side negatives, not needed
    torch.empty(1, 0, 10),  # Right-hand side negatives, not needed
)

print(scores_1)
print(scores_2)

### SNIPPET 2 ###