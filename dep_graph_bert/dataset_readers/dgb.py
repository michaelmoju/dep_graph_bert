import logging
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField, SpanField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
import spacy
import numpy as np

nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)


def parse_text(text_left, aspect, text_right):
    
    text = text_left + ' ' + aspect + ' ' + text_right
    start = len(nlp(text_left))
    end = start + len(nlp(aspect)) - 1
    doc = nlp(text)
    seq_len = len(doc)
    adj_matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in doc:
        adj_matrix[token.i][token.i] = 1
        for child in token.children:
            adj_matrix[token.i][child.i] = 1
            adj_matrix[child.i][token.i] = 1
    return doc, adj_matrix, start, end


@DatasetReader.register("dgb")
class DgbReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers
        
    def text_to_instance(self, comment_text: str, aspect: str, label: str = None) -> Instance:
        text_left, _, text_right = [s.lower() for s in comment_text.partition("$T$")]
        doc, adj_matrix, start, end = parse_text(text_left, aspect, text_right)
        text_field = TextField([Token(token.text) for token in doc], self._token_indexers)
        adj_field = ArrayField(adj_matrix)
        aspect_span_field = SpanField(start, end, text_field)
        fields = {"tokens": text_field, "adj_matrix": adj_field, "aspect_span": aspect_span_field}
        fields["meta"] = {"comment_text": comment_text, "aspect": aspect}
        
        if label:
            label_field = LabelField(label, label_namespace="labels")
            fields["label"] = label_field
        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
    
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        for i in range(0, len(lines), 3):
            comment_text = lines[i].strip()
            aspect = lines[i+1].strip()
            label = lines[i+2].strip()
            instance = self.text_to_instance(comment_text, aspect, label)
            yield instance
