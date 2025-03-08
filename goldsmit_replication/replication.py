import re
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.corpus import brown
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj
import pandas as pd
import os
current_file_path = os.path.abspath(__file__)
current_file_path = os.path.abspath(os.path.join(current_file_path, '..'))

def preprocess_text(K=1000):
    """
    Preprocesses the Brown corpus by filtering the top K words.

    Args:
        K (int): Number of most common words to keep.

    Returns:
        tuple: (word_to_idx, idx_to_word, sentences, word_indices)
    """
    words = brown.words()
    words = [word.lower() for word in words if re.match(r'^[a-zA-Z]+$', word)]
    
    counter = Counter(words)
    top_k_words = [item for item, count in counter.most_common(K)]
    word_to_idx = {word: i for i, word in enumerate(top_k_words)}
    idx_to_word = {i: word for i, word in enumerate(top_k_words)}

    sentences = brown.sents()
    sentences = [[word.lower() for word in sentence if re.match(r'^[a-zA-Z]+$', word)] for sentence in sentences]

    word_indices = defaultdict(list)
    for idx_sent, sent in enumerate(sentences):
        for idx_word, word in enumerate(sent):
            word_indices[word].append((idx_sent, idx_word))

    return word_to_idx, idx_to_word, sentences, word_indices

def compute_vicinity_vectors(top_k_words, sentences, word_indices, n_vecinity=1):
    """
    Computes the vicinity vectors for the words in the corpus.

    Args:
        top_k_words (list): List of top K words.
        sentences (list): List of sentences.
        word_indices (dict): Dictionary mapping words to their occurrences.
        n_vecinity (int): Number of words to consider in the vicinity.

    Returns:
        tuple: (vicinity_right_vector, vicinity_left_vector)
    """
    vicinity_right = {word: [] for word in top_k_words}
    vicinity_left = {word: [] for word in top_k_words}

    for top_word in tqdm(top_k_words, desc="Computing vicinity words"):
        indices = np.array(word_indices[top_word])
        for idx in indices:
            right_context = sentences[idx[0]][idx[1]+1:idx[1]+1+n_vecinity]
            left_context = sentences[idx[0]][max(idx[1]-n_vecinity, 0):idx[1]]
            vicinity_right[top_word].extend([w for w in right_context if w != top_word])
            vicinity_left[top_word].extend([w for w in left_context if w != top_word])

    vicinity_right_vector = defaultdict(list)
    vicinity_left_vector = defaultdict(list)

    for top_word in tqdm(top_k_words, desc="Building vicinity vectors"):
        counter_right = dict(Counter(vicinity_right[top_word]))
        counter_left = dict(Counter(vicinity_left[top_word]))
        
        vicinity_right_vector[top_word] = [counter_right.get(word, 0) for word in top_k_words]
        vicinity_left_vector[top_word] = [counter_left.get(word, 0) for word in top_k_words]

    return vicinity_right_vector, vicinity_left_vector

def compute_adjacency_matrix(vicinity_vector, device="mps", G=100):
    """
    Computes the adjacency matrix based on pairwise similarities.

    Args:
        vicinity_vector (dict): Dictionary of word vectors.
        device (str): Device for computation ("cpu" or "mps").
        G (int): Number of nearest neighbors.

    Returns:
        torch.Tensor: Adjacency matrix.
    """
    X = torch.tensor(np.column_stack(list(vicinity_vector.values())), dtype=torch.float32).to(device)
    norm_vectors = X.norm(dim=0, keepdim=True)
    
    pairwise_similarities = (X.T @ X) / ((norm_vectors.T) @ norm_vectors)
    pairwise_similarities.fill_diagonal_(0)
    
    _, top_indices = torch.topk(pairwise_similarities, G, dim=1)
    
    binary_mask = torch.zeros_like(pairwise_similarities, dtype=torch.float32, device=device)
    adjacency_matrix = binary_mask.scatter_(1, top_indices, 1)
    adjacency_matrix = torch.max(adjacency_matrix, adjacency_matrix.T)  # Make it symmetric
    
    return adjacency_matrix

def construct_graph(adjacency_matrix, num_nodes):
    """
    Constructs a graph from an adjacency matrix.

    Args:
        adjacency_matrix (torch.Tensor): The adjacency matrix.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        tuple: (graph, laplacian matrix)
    """
    edges = torch.nonzero(adjacency_matrix, as_tuple=False).T
    graph = Data(edge_index=edges, num_nodes=num_nodes)

    laplacian, laplacian_weights = get_laplacian(edges, normalization=None)
    laplacian_matrix = to_dense_adj(laplacian, edge_attr=laplacian_weights).squeeze(0).to("cpu")

    return graph, laplacian_matrix

def main():
    """
    Main function to execute the entire pipeline.
    """
    # Step 1: Preprocess text
    K = 1000
    word_to_idx, idx_to_word, sentences, word_indices = preprocess_text(K)
    
    # Step 2: Compute vicinity vectors
    vicinity_right_vector, vicinity_left_vector = compute_vicinity_vectors(
        list(word_to_idx.keys()), sentences, word_indices
    )
    
    # Step 3: Compute adjacency matrices
    adjacency_matrix_right = compute_adjacency_matrix(vicinity_right_vector)
    adjacency_matrix_left = compute_adjacency_matrix(vicinity_left_vector)

    # Step 4: Construct graphs
    graph_right, laplacian_right = construct_graph(adjacency_matrix_right, K)
    graph_left, laplacian_left = construct_graph(adjacency_matrix_left, K)

    # Step 5: Compute eigenvalues and eigenvectors
    eigenvalues_right, eigenvectors_right = torch.linalg.eigh(laplacian_right)
    eigenvalues_left, eigenvectors_left = torch.linalg.eigh(laplacian_left)

    # Step 6: We keep the first 2 non-constant eigenvectors (which are the 2nd and 3rd in order)
    relevant_eigenvectors_right = eigenvectors_right[:, 1:3]
    relevant_eigenvectors_left = eigenvectors_left[:, 1:3]

    # Step 7: Construct dataframe with embeddings
    df_left = pd.DataFrame(relevant_eigenvectors_right, columns = ["dim1", "dim2"])
    df_right = pd.DataFrame(relevant_eigenvectors_left, columns = ["dim1", "dim2"])
    df_left['word'] = word_to_idx.keys()
    df_right['word'] = word_to_idx.keys()
    df_left.to_csv(os.path.join(current_file_path,"embeddings_left.csv"), index=False)
    df_right.to_csv(os.path.join(current_file_path,"embeddings_right.csv"), index=False)   
    print("Graph construction completed. Eigenvalues computed.")

if __name__ == "__main__":
    main()