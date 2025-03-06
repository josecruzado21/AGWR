import os
import time
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import RGATConv
from tqdm import tqdm
import stanza
import fasttext

# Parameters
DEVICE = "cpu"
EPOCHS = 500
BATCH_SIZE = 100
LEARNING_RATE = 0.01
CHECKPOINT_INTERVAL = 100
FASTTEXT_MODEL_PATH = '.utils/fasttext/cc.en.300.bin'
STANZA_DIR = os.path.expanduser("~/stanza_resources/")
MODELS_DIR = './objects/models/'

torch.set_default_device(DEVICE)

def load_stanza_pipeline(stanza_dir):
    if os.path.exists(stanza_dir):
        return stanza.Pipeline('en', model_dir=stanza_dir, download_method=None, device=DEVICE)
    else:
        stanza.download('en')
        return stanza.Pipeline('en')

def load_fasttext_model(model_path):
    return fasttext.load_model(model_path)

def initialize_embeddings(word_to_idx, model_fasttext):
    initial_embeddings = [model_fasttext.get_word_vector(word) for word in word_to_idx]
    return torch.tensor(np.column_stack(initial_embeddings)).T

class ContextRGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = RGATConv(in_dim, hidden_dim, num_relations=3, edge_dim=1)
        self.conv2 = RGATConv(hidden_dim, out_dim, num_relations=3, edge_dim=1)
        
    def forward(self, x, edge_index, edge_type, edge_attr):
        x = self.conv1(x, edge_index, edge_type, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_type, edge_attr)
        return x

def train_model(model, optimizer, initial_embeddings, similarity_graphs, syntactic_graphs, global_cooc_edge_index, global_cooc_edge_attr, word_to_idx, idx_to_word):
    losses_track = []

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
            syn_attr = torch.ones(syn_graph.num_edges, 1, device=DEVICE)
            
            full_edge_index = torch.cat([cooc_edges, syn_edges, sim_edges], dim=1)
            full_edge_type = torch.cat([
                torch.zeros(cooc_edges.shape[1], dtype=torch.long, device=DEVICE),
                torch.ones(syn_edges.shape[1], dtype=torch.long, device=DEVICE),
                torch.full((sim_edges.shape[1],), 2, dtype=torch.long, device=DEVICE)
            ])
            full_edge_attr = torch.cat([cooc_attr, syn_attr, sim_attr])
            
            updated_embeds = model(initial_embeddings, full_edge_index, full_edge_type, full_edge_attr)
            
            mask_prob = torch.rand(len(sentence_nodes), device=DEVICE)
            mask = mask_prob < 0.15
            masked_indices = sentence_nodes[mask]

            if masked_indices.numel() == 0:
                masked_indices = sentence_nodes[torch.randint(0, len(sentence_nodes), (1,), device=DEVICE)]

            loss = F.mse_loss(updated_embeds[masked_indices], initial_embeddings[masked_indices])
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses_track.append(total_loss.item() / len(idxs_batch))
        
        end = time.time()
        print(f"Epoch {epoch+1}, Loss: {total_loss.item()/len(idxs_batch):.4f}, Time: {round(end - beginning)} seconds")
        
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, model, optimizer, losses_track)

def save_checkpoint(epoch, model, optimizer, losses_track):
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses_track
    }
    torch.save(checkpoint, os.path.join(MODELS_DIR, 'syntactic_training_checkpoint.pth'))
    torch.save(model, os.path.join(MODELS_DIR, 'syntactic_model.pth'))

def main():
    model_fasttext = load_fasttext_model(FASTTEXT_MODEL_PATH)

    cooccurrence_graph = torch.load("./objects/graphs/cooccurrence_graph.pth", weights_only=False)
    syntactic_graphs = torch.load("./objects/graphs/syntactic_graphs.pth", weights_only=False)
    similarity_graphs = torch.load("./objects/graphs/similarity_graphs.pth", weights_only=False)

    global_cooc_edge_index = cooccurrence_graph.edge_index.to(DEVICE)
    global_cooc_edge_attr = torch.ones(cooccurrence_graph.num_edges, 1, device=DEVICE)

    word_to_idx = torch.load("./objects/dictionaries/word_to_idx.pth", weights_only=False)
    idx_to_word = torch.load("./objects/dictionaries/idx_to_word.pth", weights_only=False)

    initial_embeddings = initialize_embeddings(word_to_idx, model_fasttext)

    model = ContextRGAT(300, 256, 300).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    train_model(model, optimizer, initial_embeddings, similarity_graphs, syntactic_graphs, global_cooc_edge_index, global_cooc_edge_attr, word_to_idx, idx_to_word)

if __name__ == "__main__":
    main()