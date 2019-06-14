import os
import random

"""
adapted from https://github.com/facebookresearch/PyTorch-BigGraph/blob/master/torchbiggraph/examples/livejournal.py
"""
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt',
}
TRAIN_FRACTION = 0.75

def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir


def random_split_file(fpath):
    root = os.path.dirname(fpath)

    output_paths = [
        os.path.join(root, FILENAMES['train']),
        os.path.join(root, FILENAMES['test']),
    ]
    if all(os.path.exists(path) for path in output_paths):
        print("Found some files that indicate that the input data "
              "has already been shuffled and split, not doing it again.")
        print("These files are: %s" % ", ".join(output_paths))
        return

    print('Shuffling and splitting train/test file. This may take a while.')
    train_file = os.path.join(root, FILENAMES['train'])
    test_file = os.path.join(root, FILENAMES['test'])

    print('Reading data from file: ', fpath)
    with open(fpath, "rt") as in_tf:
        lines = in_tf.readlines()

    # The first few lines are comments
    lines = lines[3:]
    print('Shuffling data')
    random.shuffle(lines)
    split_len = int(len(lines) * TRAIN_FRACTION)

    print('Splitting to train and test files')
    with open(train_file, "wt") as out_tf_train:
        for line in lines[:split_len]:
            out_tf_train.write(line)

    with open(test_file, "wt") as out_tf_test:
        for line in lines[split_len:]:
            out_tf_test.write(line)



DATA_PATH = "data/example_2/example.txt"
DATA_DIR = "data/example_2"
CONFIG_PATH = "config_2.py"

random_split_file(DATA_PATH)

### SNIPPET 1 ###

edge_paths = [os.path.join(DATA_DIR, name) for name in FILENAMES.values()]

from torchbiggraph.converters.import_from_tsv import convert_input_data

convert_input_data(
    CONFIG_PATH,
    edge_paths,
    lhs_col=0,
    rhs_col=1,
    rel_col=None,
)

### SNIPPET 2 ###

from torchbiggraph.config import parse_config
import attr
train_config = parse_config(CONFIG_PATH)

train_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['train']))]
train_config = attr.evolve(train_config, edge_paths=train_path)

from torchbiggraph.train import train
train(train_config)
# Time to run on liveJournal data: 17:43 - ???
### SNIPPET 3 ###
from torchbiggraph.eval import do_eval

eval_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['test']))]
eval_config = attr.evolve(train_config, edge_paths=eval_path)

do_eval(eval_config)

### SNIPPET 4 ###

import json
import h5py

with open(os.path.join(DATA_DIR,"dictionary.json"), "rt") as tf:
    dictionary = json.load(tf)

user_id = "0"
offset = dictionary["entities"]["user_id"].index(user_id)
print("our offset for user_id " , user_id, " is: ", offset)

with h5py.File("model/example_2/embeddings_user_id_0.v10.h5", "r") as hf:
    embedding_user_0 = hf["embeddings"][offset, :]
    embedding_all = hf["embeddings"][:]

print(embedding_all)
print(embedding_all.shape)

### SNIPPET 5 ###

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

### SNIPPET 6 ###
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
### Three common usecases: https://torchbiggraph.readthedocs.io/en/latest/downstream_tasks.html

