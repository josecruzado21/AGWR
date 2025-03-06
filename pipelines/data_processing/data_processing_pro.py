# Standard Library
import os
import re
from collections import Counter, defaultdict

# Third-Party Libraries
import numpy as np
import torch
from tqdm import tqdm
import nltk
from nltk.corpus import brown
import stanza
import fasttext
from torch_geometric.data import Data

# Custom Functions
import sys
sys.path.append(os.path.join(os.getcwd()))
from utils.helper_functions import get_syntactic_graphs, get_similarity_graphs

# Parameters
K = 25000
DEVICE = "cpu"
TEMPORAL_SLICE = 1000
N_VECINITY = 1
FASTTEXT_MODEL_PATH = './utils/fasttext/cc.en.300.bin'
STANZA_DIR = os.path.expanduser("~/stanza_resources/")
ARTIFACTS_DIR = "./objects/"

torch.set_default_device(DEVICE)

def load_fasttext_model(model_path):
    return fasttext.load_model(model_path)

def load_stanza_pipeline(stanza_dir):
    if os.path.exists(stanza_dir):
        return stanza.Pipeline('en', model_dir=stanza_dir, download_method=None, device=DEVICE)
    else:
        stanza.download('en')
        return stanza.Pipeline('en')

def preprocess_words(words):
    words_lower = [word.lower() for word in words]
    words_no_punct = [word for word in words_lower if re.match(r'^[a-zA-Z]+$', word)]
    return words_no_punct

def get_top_k_words(words, k):
    counter = Counter(words)
    top_k_words = [item for item, count in counter.most_common(k)]
    word_to_idx = dict(zip(top_k_words, range(len(top_k_words))))
    idx_to_word = dict(zip(range(len(top_k_words)), top_k_words))
    torch.save(word_to_idx, os.path.join(ARTIFACTS_DIR, "dictionaries", "word_to_idx.pth"))
    torch.save(idx_to_word, os.path.join(ARTIFACTS_DIR, "dictionaries", "idx_to_word.pth"))
    return top_k_words, word_to_idx, idx_to_word

def preprocess_sentences(sentences):
    sentences = [[word.lower() for word in sentence if re.match(r'^[a-zA-Z]+$', word)] for sentence in sentences]
    return [sentence for sentence in sentences if len(sentence) > 3]

def generate_syntactic_graphs(sentences, word_to_idx, idx_to_word, nlp):
    return get_syntactic_graphs(sentences[:TEMPORAL_SLICE], word_to_idx, idx_to_word, nlp)

def generate_cooccurrence_graph(sentences, top_k_words, word_to_idx, idx_to_word):
    word_indices = defaultdict(list)
    for idx_sent, sent in enumerate(sentences):
        for idx_word, word in enumerate(sent):
            word_indices[word].append((idx_sent, idx_word))

    vicinity_left_right = {word: [] for word in top_k_words}
    vicinity_left_right_vector = defaultdict(list)

    for top_word in tqdm(top_k_words):
        indices = np.array(word_indices[top_word])
        for idx in indices:
            right_context = sentences[idx[0]][idx[1]+1:idx[1]+1+N_VECINITY]
            left_context = sentences[idx[0]][max(idx[1]-N_VECINITY, 0):idx[1]]
            vicinity_left_right[top_word].extend([element for element in left_context + right_context if element != top_word])
        vicinity_left_right[top_word] = list(set(vicinity_left_right[top_word]))

    for top_word in tqdm(top_k_words):
        counter_left_right = dict(Counter(vicinity_left_right[top_word]))
        vicinity_left_right_vector[top_word] = [1 if counter_left_right.get(word, 0) > 0 else 0 for idx, word in idx_to_word.items()]

    source_vertices = []
    destination_vertices = []
    for word, vector in vicinity_left_right_vector.items():
        idx_word = word_to_idx[word]
        connected_idx = [i for i, value in enumerate(vector) if value == 1]
        source_vertices.extend([idx_word] * len(connected_idx))
        destination_vertices.extend(connected_idx)

    edge_index_co_currence = torch.tensor([source_vertices, destination_vertices], dtype=torch.long)
    graph_co_currence = Data(edge_index=edge_index_co_currence, num_nodes=len(word_to_idx))

    torch.save(graph_co_currence, os.path.join(ARTIFACTS_DIR, "graphs","cooccurrence_graph.pth"))

def generate_similarity_graphs(sentences, word_to_idx, syntactic_graphs, model):
    similarity_graphs = get_similarity_graphs(sentences[:TEMPORAL_SLICE], word_to_idx, syntactic_graphs, model)
    torch.save(similarity_graphs, os.path.join(ARTIFACTS_DIR, "graphs","similarity_graphs.pth"))
    return similarity_graphs

def filter_graphs(syntactic_graphs, similarity_graphs):
    idx_to_delete = []
    for i in range(len(syntactic_graphs)):
        diff = set(syntactic_graphs[i].node_index) - set(similarity_graphs[i].node_index.tolist())
        if len(diff) > 0:
            idx_to_delete.append(i)
        if (syntactic_graphs[i].edge_index.shape[0] == 0) or (similarity_graphs[i].edge_index.shape[0] == 0):
            idx_to_delete.append(i)
    syntactic_graphs = [item for i, item in enumerate(syntactic_graphs) if i not in idx_to_delete]
    similarity_graphs = [item for i, item in enumerate(similarity_graphs) if i not in idx_to_delete]

    torch.save(syntactic_graphs, os.path.join(ARTIFACTS_DIR, "graphs","syntactic_graphs.pth"))
    torch.save(similarity_graphs, os.path.join(ARTIFACTS_DIR, "graphs","similarity_graphs.pth"))

def main():
    model = load_fasttext_model(FASTTEXT_MODEL_PATH)
    nlp = load_stanza_pipeline(STANZA_DIR)

    words = brown.words()
    words_no_punct = preprocess_words(words)
    top_k_words, word_to_idx, idx_to_word = get_top_k_words(words_no_punct, K)

    sentences = brown.sents()
    sentences = preprocess_sentences(sentences)

    syntactic_graphs = generate_syntactic_graphs(sentences, word_to_idx, idx_to_word, nlp)
    generate_cooccurrence_graph(sentences, top_k_words, word_to_idx, idx_to_word)
    similarity_graphs = generate_similarity_graphs(sentences, word_to_idx, syntactic_graphs, model)
    filter_graphs(syntactic_graphs, similarity_graphs)

if __name__ == "__main__":
    main()