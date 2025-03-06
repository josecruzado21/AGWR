from tqdm import tqdm
import torch
from torch_geometric.data import Data
import numpy as np

def get_syntactic_graphs(sentences, 
                          word_to_idx_dict,
                          idx_to_word_dict,
                          stanza_object):
    syntactic_graphs = []
    non_top_N_word = []

    for element in tqdm(sentences):
        sentence = stanza_object(" ".join(element).replace(".", ""))
        
        words_index = []
        edges_index = []
        stanza_dict = {word.id: word.text for word in sentence.sentences[0].words}
        
        for word in sentence.sentences[0].words:
            word_lower = word.text.lower()
            word_idx = word_to_idx_dict.get(word_lower, -1)
            if (word_idx == -1):
                if (word_lower not in non_top_N_word):
                    non_top_N_word.append(word_lower)
            else:
                if (word_idx not in words_index):
                    words_index.append(word_idx)
                if word.head != 0:
                    head_text = stanza_dict[word.head].lower()
                    head_id = word_to_idx_dict.get(head_text, -1)
                    if head_id == -1:
                        if word_lower not in non_top_N_word: 
                            non_top_N_word.append(head_text)
                    elif word_idx in idx_to_word_dict:
                        edges_index.append((head_id, word_idx))
                        edges_index.append((word_idx, head_id))

        edge_index = torch.tensor(edges_index, 
                                  dtype=torch.long).t().contiguous()

        graph = Data(
            node_index = words_index,
            edge_index=edge_index,
        )
        syntactic_graphs.append(graph)

    return syntactic_graphs

def get_similarity_graphs(sentences, 
                          word_to_idx, 
                          syntactic_graphs_list,
                          fasttext_model):
    similarity_graphs = []
    for i, sentence in tqdm(enumerate(sentences)):
        words_idxs = []
        words_embeddings = []
        node_indexes_semantic = syntactic_graphs_list[i].node_index
        for word in sentence:
            if (word_to_idx.get(word, -1) >= 0)  & (word_to_idx.get(word, -1) not in words_idxs):
                words_idxs.append(word_to_idx.get(word))
                words_embeddings.append(fasttext_model.get_word_vector(word))
        words_idxs = torch.tensor(words_idxs)
        if len(words_idxs)>1:
            X = torch.tensor(np.column_stack(words_embeddings))
            X_norm = X.norm(dim=0, keepdim=True)
            similarity = (X.T @ X)/((X_norm.T) @ X_norm)
            similarity = similarity.fill_diagonal_(0)
            row, col = torch.nonzero(similarity, as_tuple=True)
            edge_index = torch.stack([words_idxs[row], words_idxs[col]], dim=0)
            edge_attr = similarity[row, col]
            similarity_graphs.append(Data(node_index = words_idxs,
                               edge_index=edge_index,
                                edge_attr=edge_attr))
    return similarity_graphs

def get_similarity_graphs(sentences, 
                          word_to_idx, 
                          syntactic_graphs_list,
                          fasttext_model):
    similarity_graphs = []
    for i, sentence in tqdm(enumerate(sentences)):
        words_idxs = []
        words_embeddings = []
        node_indexes_semantic = syntactic_graphs_list[i].node_index
        for word in sentence:
            if (word_to_idx.get(word, -1) >= 0)  & (word_to_idx.get(word, -1) not in words_idxs):
                words_idxs.append(word_to_idx.get(word))
                words_embeddings.append(fasttext_model.get_word_vector(word))
        words_idxs = torch.tensor(words_idxs)
        if len(words_idxs)>1:
            X = torch.tensor(np.column_stack(words_embeddings))
            X_norm = X.norm(dim=0, keepdim=True)
            similarity = (X.T @ X)/((X_norm.T) @ X_norm)
            similarity = similarity.fill_diagonal_(0)
            row, col = torch.nonzero(similarity, as_tuple=True)
            edge_index = torch.stack([words_idxs[row], words_idxs[col]], dim=0)
            edge_attr = similarity[row, col]
            similarity_graphs.append(Data(node_index = words_idxs,
                               edge_index=edge_index,
                                edge_attr=edge_attr))
    return similarity_graphs