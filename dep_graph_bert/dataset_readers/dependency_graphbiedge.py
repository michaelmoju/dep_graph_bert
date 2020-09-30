import numpy as np
import spacy
import pickle
import tqdm
nlp = spacy.load('en_core_web_sm')
import re

"""
- sentence (the target is replaced with $T$)
- target
- polarity label (0: neutral, 1:positive, -1:negative)
"""
class Example:
    def __init__(self, sentence, target, spacy_document, adj_in, adj_out, span_indices, polarity_label=None):
        self.sentence = sentence
        self.target = target
        self.spacy_document = spacy_document
        self.adj_in = adj_in
        self.adj_out = adj_out
        self.span_indices = span_indices
        self.polarity_label = polarity_label
        
def tokenize(text):
    text=text.strip()
    text=re.sub(r' {2,}',' ',text)
    document = nlp(text)
    return [token.text for token in document]

def update_edge(text,vocab):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    for token in document:
           if token.dep_ not in vocab:
               vocab[token.dep_]=len(vocab)
    return 0

def span(texts,aspect):
    startid=0
    aslen=len(tokenize(aspect))
    spans=[]
    for idx,text in enumerate(texts):
        tmp=len(tokenize(text))
        startid+=tmp
        tmp=startid
        if idx < len(texts)-1:
            startid+=aslen
            spans.append([tmp,startid])
    return spans

def dependency_adj_matrix(text,edge_vocab):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text.strip())
    seq_len = len(tokenize(text))
    adj_out = np.zeros((seq_len, seq_len)).astype('float32')
    adj_in = np.zeros((seq_len, seq_len)).astype('float32')
    edge = np.zeros((seq_len, seq_len)).astype('int32')
    edge1 = np.zeros((seq_len, seq_len)).astype('int32')
    assert len(document)==seq_len
    for token in document:
        if token.i >= seq_len:
            print('bug')
            print(text)
            print(text.split())
            print(document)
            print([token.i for token in document])
            print([token.text for token in document])
            a=input('hahha')
        if token.i < seq_len:
            adj_out[token.i][token.i] = 1
            adj_in[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    adj_out[token.i][child.i] = 1
                    adj_in[child.i][token.i] = 1
                    edge[token.i][child.i] = edge_vocab.get(child.dep_,1)
                    edge1[child.i][token.i] = edge_vocab.get(child.dep_,1)
#    print(matrix,edge)
#    a=input('hahha')
    return adj_in, adj_out, edge, edge1, document

def concat(texts,aspect):
    source=''
    splitnum=0
    for i,text in enumerate(texts):
        source+=text
        splitnum+=len(tokenize(text))
        if i <len(texts)-1:
           source+=' '+aspect+' '
           splitnum+=len(tokenize(aspect))
    if splitnum!=len(tokenize(source.strip())):
        print(texts)
        print(aspect)
        print(source)
        print(splitnum)
        print(tokenize(source.strip()))
        print(len(tokenize(source.strip())))
        a=input('gfg')
    return re.sub(r' {2,}',' ',source.strip())


def process(filename, edge_vocab=None):
    if edge_vocab is not None:
        pass
    else:
        edge_vocab={'<pad>':0,'<unk>':1}
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    
    out_examples = []
    for i in tqdm.tqdm(range(0, len(lines), 3)):
        sentence = lines[i]
        target = lines[i + 1]
        polarity_label = lines[i + 2]
        
        text_left = [s.lower().strip() for s in sentence.split("$T$")]
        aspect = target.lower().strip()
        span_indices = span(text_left, aspect)
        adj_in, adj_out, edge, edge1, document = dependency_adj_matrix(concat(text_left,aspect),edge_vocab)
        
        out_examples.append(Example(sentence, target, document, adj_in, adj_out, span_indices, polarity_label))
    return out_examples
