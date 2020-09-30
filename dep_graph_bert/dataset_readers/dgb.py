import logging
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField, SpanField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from .dependency_graphbiedge import text_to_example, Example
from tqdm import tqdm

logger = logging.getLogger(__name__)


@DatasetReader.register("dgb")
class DgbReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers
        
    def text_to_instance(self, sentence: str, target: str, polarity_label: str = None) -> Instance:
        example: Example = text_to_example(sentence, target, polarity_label)
        text_field = TextField([Token(token.text) for token in example.spacy_document], self._token_indexers)
        adj_in_field = ArrayField(example.adj_in)
        adj_out_field = ArrayField(example.adj_out)
        transformer_indices = ArrayField(example.transformer_indices)
        span_indices = ArrayField(example.span_indices)
        
        fields = {"tokens": text_field, "adj_in": adj_in_field, "adj_out": adj_out_field,
                  "transformer_indices": transformer_indices, "span_indices": span_indices}
        if example.polarity_label:
            label_field = LabelField(polarity_label, label_namespace="labels")
            fields["label"] = label_field
        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
    
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        for i in tqdm(range(0, len(lines), 3)):
            comment_text = lines[i].strip()
            aspect = lines[i+1].strip()
            label = lines[i+2].strip()
            instance = self.text_to_instance(comment_text, aspect, label)
            yield instance
