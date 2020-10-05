# make necessary directories
mkdir "trainedmodels"

# set variables
PACKAGE_PATH=$(realpath "$(pwd)")
echo "package file path: $PACKAGE_PATH"
PYTHONPATH="$PACKAGE_PATH:$PYTHONPATH"

export PYTHONPATH
export TRAIN_DATA_PATH="/work/data/acl-14-short-data/train.raw"
export TRAIN_DATA_PATH="/work/data/acl-14-short-data/test.raw"
export DEV_DATA_PATH="/work/data/acl-14-short-data/test.raw"

# train model with allennlp
allennlp train dep_graph_bert.jsonnet -s trainedmodels -f --include-package dep_graph_bert