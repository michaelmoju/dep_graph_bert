local BERT_MODEL_NAME = "bert-base-uncased";
local glove_file = "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz";
local train_path = std.extVar("TRAIN_DATA_PATH");
local val_path = std.extVar("DEV_DATA_PATH");

{
  "dataset_reader": {
    "type": "dgb",
    "token_indexers": {
      "word_pieces": {
        "type": "pretrained_transformer_mismatched",
        "model_name": BERT_MODEL_NAME,
        "namespace": "tags",
        "max_length": 512
      }
     }
  },
  "data_loader": {
    "type": "default",
    "batch_size": 16,
    "shuffle": true
  },
  "train_data_path": train_path,
  "validation_data_path": val_path,
  "model": {
    "type": "asbigcn",
    "text_field_embedder": {
      "type": "basic",
      "token_embedders": {
        "word_pieces": {
            "type": "pretrained_transformer_mismatchd",
            "model_name": BERT_MODEL_NAME,
            "max_length": 512
        }
      }
    }
  },
  "trainer": {
    "type": "gradient_descent",
    "optimizer": {
        "type": "adam",
        "lr": 1e-5
    },
    "num_epochs": 10,
    "grad_clipping": 1.0,
    "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "mode": "max",
        "patience": 0
    },
    "cuda_device": 0
  }
}
