import os
import random
import json
import h5py
import attr

from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.config import parse_config
from torchbiggraph.train import train
from torchbiggraph.eval import do_eval

"""
adapted from https://github.com/facebookresearch/PyTorch-BigGraph/blob/master/torchbiggraph/examples/livejournal.py
"""
# ----------------------------------------------------------------------------------------------------------------------
# Helper functions, and constants

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
    lines = lines[4:]
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

DATA_PATH = "data/example_1/example.txt"
DATA_DIR = "data/example_1"
CONFIG_PATH = "config_1.py"
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt',
}
TRAIN_FRACTION = 0.75

edge_paths = [os.path.join(DATA_DIR, name) for name in FILENAMES.values()]
train_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['train']))]
eval_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['test']))]

# ----------------------------------------------------------------------------------------------------------------------
#

def run_train_eval():
    random_split_file(DATA_PATH)

    convert_input_data(
        CONFIG_PATH,
        edge_paths,
        lhs_col=0,
        rhs_col=1,
        rel_col=None,
    )

    train_config = parse_config(CONFIG_PATH)

    train_config = attr.evolve(train_config, edge_paths=train_path)

    train(train_config)

    eval_config = attr.evolve(train_config, edge_paths=eval_path)

    do_eval(eval_config)

def output_embedding():
    with open(os.path.join(DATA_DIR, "dictionary.json"), "rt") as tf:
        dictionary = json.load(tf)

    user_id = "0"
    offset = dictionary["entities"]["user_id"].index(user_id)
    print("our offset for user_id ", user_id, " is: ", offset)

    with h5py.File("model/example_1/embeddings_user_id_0.v10.h5", "r") as hf:
        embedding = hf["embeddings"][offset, :]

    print(f" our embedding looks like this: {embedding}")
    print(f"and has a size of: {embedding.shape}")

# ----------------------------------------------------------------------------------------------------------------------
# Main method

def main():
    run_train_eval()

    output_embedding()

if __name__ == "__main__":
    main()