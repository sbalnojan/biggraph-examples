
from torchbiggraph.config import parse_config
import attr
train_config = parse_config(CONFIG_PATH)

train_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['train']))]
train_config = attr.evolve(train_config, edge_paths=train_path)

from torchbiggraph.train import train
train(train_config)

# Time to run on liveJournal data: 17:43 - ???
### SNIPPET 3 ###