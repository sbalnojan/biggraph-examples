
from torchbiggraph.eval import do_eval

eval_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['test']))]
eval_config = attr.evolve(train_config, edge_paths=eval_path)

do_eval(eval_config)

### SNIPPET 4 ###