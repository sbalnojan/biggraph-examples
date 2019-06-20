
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