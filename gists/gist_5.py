import json
import h5py

with open(os.path.join(DATA_DIR,"dictionary.json"), "rt") as tf:
    dictionary = json.load(tf)

user_id = "0"
offset = dictionary["entities"]["user_id"].index(user_id)
print("our offset for user_id " , user_id, " is: ", offset)

with h5py.File("model/example_1/embeddings_user_id_" + user_id + ".v10.h5", "r") as hf:
    embedding = hf["embeddings"][0, :]

print(embedding)
print(embedding.shape)
### SNIPPET 5 ###