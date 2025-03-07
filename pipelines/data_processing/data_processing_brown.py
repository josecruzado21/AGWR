import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)

import re
from collections import Counter, defaultdict
import numpy as np
import torch
from tqdm import tqdm
from nltk.corpus import brown
from torch_geometric.data import Data
import yaml

# Custom Functions
from utils.helper_functions import (get_syntactic_graphs, 
                                    get_similarity_graphs, 
                                    load_fasttext_model, 
                                    load_stanza_pipeline)

# Parameters
FASTTEXT_MODEL_PATH = './utils/fasttext/cc.en.300.bin'
STANZA_DIR = os.path.expanduser("~/stanza_resources/")
ARTIFACTS_DIR = "./objects/"
with open(os.path.join(root_path, "parameters","parameters.yaml"), "r") as file:
    params = yaml.safe_load(file)
K = params["K"]
N_SENTENCES = params["N_SENTENCES"]
N_VECINITY = params["N_VECINITY"]


def preprocess_words(words):
    """
        Load the Stanza pipeline.

    Args:
        stanza_dir (str): Directory where the Stanza models are stored.

    Returns:
        stanza.Pipeline: Loaded Stanza pipeline.
    """
    words_lower = [word.lower() for word in words]
    words_no_punct = [word for word in words_lower if re.match(r'^[a-zA-Z]+$', word)]
    return words_no_punct

def get_top_k_words_brown(words, k):
    """
    Load the Stanza pipeline.

    Args:
        stanza_dir (str): Directory where the Stanza models are stored.

    Returns:
        stanza.Pipeline: Loaded Stanza pipeline.
    """
    counter = Counter(words)
    top_k_words = [item for item, count in counter.most_common(k)]
    word_to_idx = dict(zip(top_k_words, range(len(top_k_words))))
    idx_to_word = dict(zip(range(len(top_k_words)), top_k_words))
    torch.save(word_to_idx, os.path.join(ARTIFACTS_DIR, "dictionaries", "word_to_idx_brown.pth"))
    torch.save(idx_to_word, os.path.join(ARTIFACTS_DIR, "dictionaries", "idx_to_word_brown.pth"))
    return top_k_words, word_to_idx, idx_to_word

def preprocess_sentences(sentences):
    """
    Preprocess sentences by converting words to lowercase and removing punctuation.

    Args:
        sentences (list of list of str): List of sentences, where each sentence is a list of words.

    Returns:
        list: List of preprocessed sentences.
    """
    sentences = [[word.lower() for word in sentence if re.match(r'^[a-zA-Z]+$', word)] for sentence in sentences]
    return [sentence for sentence in sentences if len(sentence) > 3]

def generate_cooccurrence_graph(sentences, top_k_words, word_to_idx, idx_to_word):
    """
    Generate a co-occurrence graph from sentences.

    Args:
        sentences (list of list of str): List of sentences, where each sentence is a list of words.
        top_k_words (list of str): List of top K words.
        word_to_idx (dict): Dictionary mapping words to their indices.
        idx_to_word (dict): Dictionary mapping indices to their words.

    Returns:
        Data: Co-occurrence graph.
    """
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

    torch.save(graph_co_currence, os.path.join(ARTIFACTS_DIR, "graphs","cooccurrence_graph_brown.pth"))

def generate_similarity_graphs(sentences, word_to_idx, model):
    """
    Generate similarity graphs from sentences.

    Args:
        sentences (list of list of str): List of sentences, where each sentence is a list of words.
        word_to_idx (dict): Dictionary mapping words to their indices.
        model (FastText model): FastText model for obtaining word vectors.

    Returns:
        list: List of similarity graphs.
    """
    similarity_graphs = get_similarity_graphs(sentences[:N_SENTENCES], word_to_idx, model)
    torch.save(similarity_graphs, os.path.join(ARTIFACTS_DIR, "graphs","similarity_graphs_brown.pth"))
    return similarity_graphs

def filter_graphs(syntactic_graphs, similarity_graphs):
    """
    Filter out graphs with missing nodes or empty edges.

    Args:
        syntactic_graphs (list): List of syntactic graphs.
        similarity_graphs (list): List of similarity graphs.

    Returns:
        tuple: Filtered lists of syntactic and similarity graphs.
    """
    idx_to_delete = []
    for i in range(len(syntactic_graphs)):
        diff = set(syntactic_graphs[i].node_index) - set(similarity_graphs[i].node_index.tolist())
        if len(diff) > 0:
            idx_to_delete.append(i)
        if (syntactic_graphs[i].edge_index.shape[0] == 0) or (similarity_graphs[i].edge_index.shape[0] == 0):
            idx_to_delete.append(i)
    syntactic_graphs = [item for i, item in enumerate(syntactic_graphs) if i not in idx_to_delete]
    similarity_graphs = [item for i, item in enumerate(similarity_graphs) if i not in idx_to_delete]

    torch.save(syntactic_graphs, os.path.join(ARTIFACTS_DIR, "graphs","syntactic_graphs_brown.pth"))
    torch.save(similarity_graphs, os.path.join(ARTIFACTS_DIR, "graphs","similarity_graphs_brown.pth"))

def main():
    model = load_fasttext_model(FASTTEXT_MODEL_PATH)
    nlp = load_stanza_pipeline(STANZA_DIR)

    words = brown.words()
    words_no_punct = preprocess_words(words)
    top_k_words, word_to_idx, idx_to_word = get_top_k_words_brown(words_no_punct, K)

    sentences = brown.sents()
    sentences = preprocess_sentences(sentences)

    syntactic_graphs = get_syntactic_graphs(sentences[:N_SENTENCES], 
                                            word_to_idx,
                                            idx_to_word, 
                                            nlp,
                                            progress_bar=True)
    generate_cooccurrence_graph(sentences, top_k_words, word_to_idx, idx_to_word)
    similarity_graphs = generate_similarity_graphs(sentences, word_to_idx, model)
    filter_graphs(syntactic_graphs, similarity_graphs)

if __name__ == "__main__":
    main()