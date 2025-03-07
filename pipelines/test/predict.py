import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)


import torch
import stanza
import fasttext
import numpy as np
from torch_geometric.nn import RGATConv
from pipelines.train.brown.training_brown import ContextRGAT
from utils.helper_functions import get_syntactic_graphs, get_similarity_graphs, load_fasttext_model, load_stanza_pipeline

# Constants
STANZA_DIR = os.path.expanduser("~/stanza_resources/")
FASTTEXT_MODEL_PATH = './utils/fasttext/cc.en.300.bin'
COOCCURRENCE_GRAPH_PATH = "./objects/graphs/cooccurrence_graph.pth"
SIMILARITY_GRAPHS_PATH = "./objects/graphs/similarity_graphs.pth"
WORD_TO_IDX_PATH = "./objects/dictionaries/word_to_idx.pth"
IDX_TO_WORD_PATH = "./objects/dictionaries/idx_to_word.pth"
MODEL_PATH = './objects/models/syntactic_model.pth'
PREDICTIONS_PATH = './objects/predictions/predictions.pth'

def predict_embeddings(sentences, cooccurrence_graph, word_to_idx, idx_to_word, model, nlp, model_fasttext):
    """
    Predict embeddings for given sentences using the trained model.
    
    Args:
        sentences (list): List of sentences to predict embeddings for.
        cooccurrence_graph (Data): Co-occurrence graph data.
        word_to_idx (dict): Dictionary mapping words to indices.
        trained_model (torch.nn.Module): Trained model for prediction.
        nlp (stanza.Pipeline): NLP pipeline for syntactic parsing.
        model_fasttext (fasttext.FastText._FastText): FastText model for word embeddings.
    
    Returns:
        tuple: Predicted embeddings dictionary and list.
    """
    # Prepare global co-occurrence data
    global_cooc_edge_index = cooccurrence_graph.edge_index

    # Initial embeddings
    initial_embeddings = torch.tensor(np.column_stack([model_fasttext.get_word_vector(word) for word in word_to_idx]).T)

    predicted_embeddings_dict = []
    predicted_embeddings_list = []
    syntactic_graph_test = get_syntactic_graphs(sentences, word_to_idx, idx_to_word, nlp)
    similarity_graph_test = get_similarity_graphs(sentences, word_to_idx, model_fasttext)
    
    for i in range(len(sentences)):
        sentence_nodes = torch.tensor([word_to_idx[w] for w in sentences[i] if w in word_to_idx])
        test_sy_graph = syntactic_graph_test[i]
        test_sim_graph = similarity_graph_test[i]
        syntactic_edge_index = test_sy_graph.edge_index
        sim_edge_index = test_sim_graph.edge_index
        sim_edge_attr = test_sim_graph.edge_attr
        
        # Filter global co-occurrence edges
        mask = torch.isin(global_cooc_edge_index, sentence_nodes).all(dim=0)
        cooc_edges = global_cooc_edge_index[:, mask]
        cooc_attr = torch.ones(cooc_edges.shape[1], 1)

        # Combine all edges
        full_edge_index = torch.cat([cooc_edges, syntactic_edge_index, sim_edge_index], dim=1)
        full_edge_type = torch.cat([
            torch.zeros(cooc_edges.shape[1], dtype=torch.long),
            torch.ones(syntactic_edge_index.shape[1], dtype=torch.long),
            torch.full((sim_edge_index.shape[1],), 2)
        ])
        full_edge_attr = torch.cat([cooc_attr, torch.ones(syntactic_edge_index.shape[1], 1), sim_edge_attr.unsqueeze(1)])
        
        model.eval()
        with torch.no_grad():
            updated_embeds = model(initial_embeddings, full_edge_index, full_edge_type, full_edge_attr)

            # Extract sentence-specific embeddings
            sentence_embeddings = updated_embeds[sentence_nodes]
        words_str = [idx_to_word[i.item()] for i in sentence_nodes]
        predicted_embeddings_list.append(sentence_embeddings)
        predicted_embeddings_dict.append(dict(zip(words_str, sentence_embeddings)))
    
    return predicted_embeddings_dict, predicted_embeddings_list

if __name__ == "__main__":
    # Example usage
    nlp = load_stanza_pipeline(STANZA_DIR)
    model_fasttext = load_fasttext_model(FASTTEXT_MODEL_PATH)
    cooccurrence_graph = torch.load("./objects/graphs/cooccurrence_graph.pth", weights_only=False)
    syntactic_graphs = torch.load("./objects/graphs/syntactic_graphs.pth", weights_only=False)
    similarity_graphs = torch.load("./objects/graphs/similarity_graphs.pth", weights_only=False)
    word_to_idx = torch.load("./objects/dictionaries/word_to_idx.pth", weights_only=False)
    idx_to_word = torch.load("./objects/dictionaries/idx_to_word.pth", weights_only=False)
    model = torch.load(MODEL_PATH, weights_only=False)
    sentences = [['order', 'a', 'work', 'stoppage'], ['order', 'me', 'some', 'flowers']]
    predicted_embeddings_dict, predicted_embeddings_list = predict_embeddings(
        sentences,
        cooccurrence_graph, word_to_idx,idx_to_word,model, nlp, model_fasttext
    )
    
    # Save predictions
    torch.save({
        'predicted_embeddings_dict': predicted_embeddings_dict,
        'predicted_embeddings_list': predicted_embeddings_list
    }, PREDICTIONS_PATH)
    
    print(f"Predictions saved to {PREDICTIONS_PATH}")