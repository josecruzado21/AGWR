# Standard Library
import os
import re
from collections import Counter, defaultdict
import numpy as np
import torch
from tqdm import tqdm
# NLP Libraries
import nltk
from nltk.corpus import brown
import stanza
import fasttext
import fasttext.util
# Graph Processing
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils  # Aliasing for clarity
# Custom Functions
import sys
sys.path.append(os.path.join(os.getcwd(), "utils"))
from utils import get_syntactic_graphs, get_similarity_graphs
model = fasttext.load_model('./fasttext/cc.en.300.bin')

# Parameters
K=250
device = "cpu"
temporal_slice = 10
torch.set_default_device(device)

# Load syntactic parser

stanza_dir = os.path.expanduser("~/stanza_resources/")
if os.path.exists(stanza_dir):
    nlp = stanza.Pipeline('en', model_dir=stanza_dir, download_method=None, device = "cpu")
else:
    stanza.download('en') # download English model
    nlp = stanza.Pipeline('en') # initialize English neural pipeline

# Processing

words = brown.words()
words_lower = [word.lower() for word in words]

# Most common words
words_no_punct = [word for word in words_lower if re.match(r'^[a-zA-Z]+$', word)]
counter = Counter(words_no_punct)
top_k_words = [item for item, count in counter.most_common(K)]
word_to_idx = dict(zip(top_k_words, range(len(top_k_words))))
idx_to_word = dict(zip(range(len(top_k_words)), top_k_words))
torch.save(word_to_idx, "./artifacts/word_to_idx.pth")
torch.save(idx_to_word, "./artifacts/idx_to_word.pth")

sentences = brown.sents()
sentences = [[word.lower() for word in sentence if re.match(r'^[a-zA-Z]+$', word)] for sentence in sentences]
sentences = [sentence for sentence in sentences if len(sentence)>3]

# Syntactic graph generation
syntactic_graphs = get_syntactic_graphs(sentences[0: temporal_slice],
                                         word_to_idx,
                                         idx_to_word,
                                         nlp)

# Co-occurrence graph generation
# Then we determine the indexes of the sentence and word where each word appears
# This will be a dictionary with a key per word, where the value a is a list with all pairs of
# index of sentence and index of word within the sentence where each word appears
word_indices = defaultdict(list)
for idx_sent, sent in enumerate(sentences):
    for idx_word, word in enumerate(sent):
            word_indices[word].append((idx_sent, idx_word))

# # Calculate vecinity_right efficiently
n_vecinity = 1
vicinity_left_right = {word: [] for word in top_k_words}
vicinity_left_right_vector = defaultdict(list)

for top_word in tqdm(top_k_words):
    indices = np.array(word_indices[top_word])
    for idx in indices:
        right_context = sentences[idx[0]][idx[1]+1:idx[1]+1+n_vecinity]
        left_context = sentences[idx[0]][max(idx[1]-n_vecinity, 0):idx[1]]
        vicinity_left_right[top_word].extend([element for element in left_context + 
                                              right_context if element!=top_word])
    vicinity_left_right[top_word] = list(set(vicinity_left_right[top_word]))
    
    
for top_word in tqdm(top_k_words):
    counter_left_right = dict(Counter(vicinity_left_right[top_word]))
    vicinity_left_right_vector[top_word] = [1 if counter_left_right.get(word, 0) > 0 
                                       else 0 for idx, word in idx_to_word.items()]
    
# Now create graph
source_vertices = []
destination_vertices = []
for word, vector in vicinity_left_right_vector.items():
    idx_word = word_to_idx[word]
    connected_idx = [i for i, value in enumerate(vector) if value == 1]
    source_vertices.extend([idx_word]*len(connected_idx))
    destination_vertices.extend(connected_idx)
    
edge_index_co_currence = torch.tensor([source_vertices, destination_vertices], dtype=torch.long)
graph_co_currence = Data(
    edge_index=edge_index_co_currence,
    num_nodes=len(word_to_idx)
)

torch.save(graph_co_currence, "./artifacts/graphs/cooccurrence_graph.pth")

# Similarity graph generation
similarity_graphs = get_similarity_graphs(sentences[0:temporal_slice],
                                         word_to_idx,
                                         syntactic_graphs,
                                         model)
torch.save(similarity_graphs, "./artifacts/graphs/similarity_graphs.pth")

idx_to_delete = []
for i in range(len(syntactic_graphs)):
    diff = set(syntactic_graphs[i].node_index) - set(similarity_graphs[i].node_index.tolist())
    if len(diff)>0:
        idx_to_delete.append(i)
    if (syntactic_graphs[i].edge_index.shape[0] == 0) | (similarity_graphs[i].edge_index.shape[0] == 0):
        idx_to_delete.append(i)
syntactic_graphs = [item for i, item in enumerate(syntactic_graphs) if i not in idx_to_delete]
similarity_graphs = [item for i, item in enumerate(similarity_graphs) if i not in idx_to_delete]

torch.save(syntactic_graphs, "./artifacts/graphs/syntactic_graphs.pth")
torch.save(similarity_graphs, "./artifacts/graphs/similarity_graphs.pth")