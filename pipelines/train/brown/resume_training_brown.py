
import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../../..'))
sys.path.append(root_path)
from utils.helper_functions import (load_checkpoint, 
                                    save_checkpoint, 
                                    load_fasttext_model,
                                    initialize_embeddings)
import yaml
import torch
import random
import time
import torch.nn.functional as F

# Parameters
FASTTEXT_MODEL_PATH = './utils/fasttext/cc.en.300.bin'
STANZA_DIR = os.path.expanduser("~/stanza_resources/")
MODELS_DIR = './objects/models/'
with open(os.path.join(root_path, "parameters","parameters.yaml"), "r") as file:
    params = yaml.safe_load(file)
EPOCHS = params["EPOCHS"]
BATCH_SIZE = params["BATCH_SIZE"]
LEARNING_RATE = params["LEARNING_RATE"]
CHECKPOINT_INTERVAL = params["CHECKPOINT_INTERVAL"]

def train_model_resume(model, 
                optimizer, 
                initial_embeddings, 
                similarity_graphs, 
                syntactic_graphs,
                global_cooc_edge_index, 
                global_cooc_edge_attr,
                losses,
                initial_epoch):
    """
    Resume training the ContextRGAT model from a checkpoint.

    Args:
        model (ContextRGAT): The ContextRGAT model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        initial_embeddings (torch.Tensor): Initial embeddings for the nodes.
        similarity_graphs (list): List of similarity graphs.
        syntactic_graphs (list): List of syntactic graphs.
        global_cooc_edge_index (torch.Tensor): Edge indices for the global co-occurrence graph.
        global_cooc_edge_attr (torch.Tensor): Edge attributes for the global co-occurrence graph.
        losses (list): List of tracked losses from previous training.
        initial_epoch (int): The epoch number to resume training from.

    Returns:
        None
    """
    losses_track = losses

    for epoch in range(EPOCHS):
        idxs_batch = random.sample(range(len(similarity_graphs)), BATCH_SIZE)
        beginning = time.time()
        total_loss = 0

        optimizer.zero_grad()  # Reset gradients before batch
        
        for sent_idx in idxs_batch:
            syn_graph = syntactic_graphs[sent_idx]
            sim_graph = similarity_graphs[sent_idx]
            
            sentence_nodes = sim_graph.node_index
            
            mask = (global_cooc_edge_index.unsqueeze(-1) == sentence_nodes).any(dim=-1).all(dim=0)
            cooc_edges = global_cooc_edge_index[:, mask]
            cooc_attr = global_cooc_edge_attr[mask]
            
            sim_edges = sim_graph.edge_index
            sim_attr = sim_graph.edge_attr.unsqueeze(1)
            
            syn_edges = syn_graph.edge_index
            syn_attr = torch.ones(syn_graph.num_edges, 1)
            
            full_edge_index = torch.cat([cooc_edges, syn_edges, sim_edges], dim=1)
            full_edge_type = torch.cat([
                torch.zeros(cooc_edges.shape[1], dtype=torch.long),
                torch.ones(syn_edges.shape[1], dtype=torch.long),
                torch.full((sim_edges.shape[1],), 2, dtype=torch.long)
            ])
            full_edge_attr = torch.cat([cooc_attr, syn_attr, sim_attr])
            
            updated_embeds = model(initial_embeddings, full_edge_index, full_edge_type, full_edge_attr)
            
            mask_prob = torch.rand(len(sentence_nodes))
            mask = mask_prob < 0.15
            masked_indices = sentence_nodes[mask]

            if masked_indices.numel() == 0:
                masked_indices = sentence_nodes[torch.randint(0, len(sentence_nodes), (1,))]

            loss = F.mse_loss(updated_embeds[masked_indices], initial_embeddings[masked_indices])
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses_track.append(total_loss.item() / len(idxs_batch))
        
        end = time.time()
        print(f"Epoch {initial_epoch+epoch+1}, Loss: {total_loss.item()/len(idxs_batch):.4f}, Time: {round(end - beginning)} seconds")
        
        if (initial_epoch + epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, model, optimizer, losses_track, "brown")

def main():
    model_fasttext = load_fasttext_model(FASTTEXT_MODEL_PATH)

    cooccurrence_graph = torch.load("./objects/graphs/cooccurrence_graph.pth", weights_only=False)
    syntactic_graphs = torch.load("./objects/graphs/syntactic_graphs.pth", weights_only=False)
    similarity_graphs = torch.load("./objects/graphs/similarity_graphs.pth", weights_only=False)

    global_cooc_edge_index = cooccurrence_graph.edge_index
    global_cooc_edge_attr = torch.ones(cooccurrence_graph.num_edges, 1)

    word_to_idx = torch.load("./objects/dictionaries/word_to_idx.pth", weights_only=False)

    initial_embeddings = initialize_embeddings(word_to_idx, model_fasttext)

    model, optimizer, losses_track, epochs = load_checkpoint("brown")

    model.train()
    train_model_resume(model, optimizer, 
                       initial_embeddings, similarity_graphs, 
                       syntactic_graphs, global_cooc_edge_index, 
                       global_cooc_edge_attr, losses_track, epochs)

if __name__ == "__main__":
    main()