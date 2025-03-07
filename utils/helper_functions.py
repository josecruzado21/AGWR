import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '..'))
sys.path.append(root_path)

from tqdm import tqdm
import torch
from torch_geometric.data import Data
import numpy as np
import stanza
import fasttext
from collections import Counter
from utils.ContextRGAT import ContextRGAT
ARTIFACTS_DIR = "./objects/"
MODELS_DIR = "./objects/models/"

def get_syntactic_graphs(sentences, 
                          word_to_idx_dict,
                          idx_to_word_dict,
                          stanza_object,
                          progress_bar = False):
    """
    Generate syntactic graphs from sentences.

    Args:
        sentences (list of list of str): List of sentences, where each sentence is a list of words.
        word_to_idx_dict (dict): Dictionary mapping words to their indices.
        idx_to_word_dict (dict): Dictionary mapping indices to their words.
        stanza_object (Stanza object): Stanza NLP object for processing sentences.
        progress_bar (bool, optional): Whether to display a progress bar. Defaults to False.

    Returns:
        list: List of syntactic graphs.
    """
    syntactic_graphs = []
    non_top_N_word = []
    if progress_bar:
        sentences_iterator = tqdm(sentences) # Use tqdm for progress bar if enabled
    else:
        sentences_iterator = sentences
    for element in sentences_iterator:
        # Process the sentence with Stanza
        sentence = stanza_object(" ".join(element).replace(".", ""))
        
        words_index = []
        edges_index = []

        # Create a dictionary mapping word IDs to their text
        stanza_dict = {word.id: word.text for word in sentence.sentences[0].words}
        
        for word in sentence.sentences[0].words:
            word_lower = word.text.lower()
            word_idx = word_to_idx_dict.get(word_lower, -1)
            if (word_idx == -1):
                # If the word is not in the dictionary, add it to non_top_N_word list
                if (word_lower not in non_top_N_word):
                    non_top_N_word.append(word_lower)
            else:
                # If the word is in the dictionary, add its index to words_index
                if (word_idx not in words_index):
                    words_index.append(word_idx)
                if word.head != 0:
                    # If the word has a head, find the head's index
                    head_text = stanza_dict[word.head].lower()
                    head_id = word_to_idx_dict.get(head_text, -1)
                    if head_id == -1:
                        # If the head is not in the dictionary, add it to non_top_N_word list
                        if word_lower not in non_top_N_word: 
                            non_top_N_word.append(head_text)
                    elif word_idx in idx_to_word_dict:
                        # Add edges between the word and its head
                        edges_index.append((head_id, word_idx))
                        edges_index.append((word_idx, head_id))

        # Create edge_index tensor from edges_index list
        edge_index = torch.tensor(edges_index, 
                                  dtype=torch.long).t().contiguous()

        # Create a graph with node indices and edge indices
        graph = Data(
            node_index = words_index,
            edge_index=edge_index,
        )
        syntactic_graphs.append(graph)

    return syntactic_graphs

def get_similarity_graphs(sentences, 
                          word_to_idx, 
                          fasttext_model):
    """
    Generate similarity graphs from sentences.

    Args:
        sentences (list of list of str): List of sentences, where each sentence is a list of words.
        word_to_idx (dict): Dictionary mapping words to their indices.
        fasttext_model (FastText model): FastText model for obtaining word vectors.

    Returns:
        list: List of similarity graphs.
    """
    similarity_graphs = []
    for i, sentence in enumerate(sentences):
        words_idxs = []
        words_embeddings = []
        for word in sentence:
            # Check if the word is in the dictionary and not already added
            if (word_to_idx.get(word, -1) >= 0)  & (word_to_idx.get(word, -1) not in words_idxs):
                words_idxs.append(word_to_idx.get(word))
                words_embeddings.append(fasttext_model.get_word_vector(word))
        words_idxs = torch.tensor(words_idxs)
        if len(words_idxs)>1:
            # Create a tensor of word embeddings
            X = torch.tensor(np.column_stack(words_embeddings))
            # Normalize the embeddings
            X_norm = X.norm(dim=0, keepdim=True)
            # Compute the similarity matrix
            similarity = (X.T @ X)/((X_norm.T) @ X_norm)
            # Set the diagonal to zero to remove self-loops
            similarity = similarity.fill_diagonal_(0)
            # Get the indices of non-zero similarities
            row, col = torch.nonzero(similarity, as_tuple=True)
            # Create edge_index tensor from the similarity indices
            edge_index = torch.stack([words_idxs[row], words_idxs[col]], dim=0)
            # Get the similarity values for the edges
            edge_attr = similarity[row, col]
            # Create a graph with node indices, edge indices, and edge attributes
            similarity_graphs.append(Data(node_index = words_idxs,
                               edge_index=edge_index,
                                edge_attr=edge_attr))
    return similarity_graphs

def load_stanza_pipeline(stanza_dir):
    """
    Load the Stanza pipeline.

    Args:
        stanza_dir (str): Directory where the Stanza models are stored.

    Returns:
        stanza.Pipeline: Loaded Stanza pipeline.
    """
   
    if os.path.exists(stanza_dir):
        return stanza.Pipeline('en', model_dir=stanza_dir, download_method=None)
    else:
        stanza.download('en')
        return stanza.Pipeline('en')
    
def load_fasttext_model(model_path):
    """
    Load the FastText model.

    Args:
        model_path (str): Path to the FastText model file.

    Returns:
        fasttext.FastText._FastText: Loaded FastText model.
    """
    return fasttext.load_model(model_path)

def initialize_embeddings(word_to_idx, model_fasttext):
    """
    Initialize embeddings for words using a FastText model.

    Args:
        word_to_idx (dict): Dictionary mapping words to their indices.
        model_fasttext (FastText model): FastText model for obtaining word vectors.

    Returns:
        torch.Tensor: Tensor containing the initialized embeddings.
    """
    initial_embeddings = [model_fasttext.get_word_vector(word) for word in word_to_idx]
    return torch.tensor(np.column_stack(initial_embeddings)).T

def save_checkpoint(epoch, model, optimizer, losses_track, model_type):
    """
    Save the model checkpoint.

    Args:
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer to be saved.
        losses_track (list): List of tracked losses.
        model_type (str): Type of the model.

    Returns:
        None
    """
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses_track
    }
    torch.save(checkpoint, os.path.join(MODELS_DIR, f'{model_type}',f'{model_type}_checkpoint.pth'))
    torch.save(model, os.path.join(MODELS_DIR, f'{model_type}', f'{model_type}_model.pth'))

def load_checkpoint(model_type):
    """
    Load the model checkpoint.

    Args:
        model_type (str): Type of the model.

    Returns:
        tuple: Loaded model, optimizer, list of tracked losses, and current epoch number.
    """
    model = torch.load(os.path.join(MODELS_DIR, f"{model_type}",f'{model_type}_model_base.pth'),weights_only=False)
    optimizer = torch.load(os.path.join(MODELS_DIR, f"{model_type}",f'{model_type}_optimizer_base.pth'), weights_only=False)
    checkpoint = torch.load(os.path.join(MODELS_DIR, f"{model_type}",f'{model_type}_checkpoint.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    losses_track = checkpoint['losses']
    epochs = checkpoint['epoch']
    return model, optimizer, losses_track, epochs