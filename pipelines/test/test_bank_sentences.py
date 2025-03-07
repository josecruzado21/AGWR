import os
import sys
import re
import torch
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)
from pipelines.train.brown.training_brown import ContextRGAT
from utils.helper_functions import load_stanza_pipeline, load_fasttext_model
from pipelines.test.predict import predict_embeddings
import pandas as pd

with open(os.path.join(root_path, "data","predict", "bank_sentences", "bank_sentences.txt"), "r") as file:
    sentences = file.readlines()
    sentences = [sentence.replace(".", "").replace(",", "").replace("\n", "").lower().split(" ")
                 for sentence in sentences]
    labels = [1]*50 + [0]*50

def main():
    STANZA_DIR = os.path.expanduser("~/stanza_resources/")
    FASTTEXT_MODEL_PATH = './utils/fasttext/cc.en.300.bin'
    MODEL_PATH = './objects/models/brown/brown_model.pth'
    nlp = load_stanza_pipeline(STANZA_DIR)
    model_fasttext = load_fasttext_model(FASTTEXT_MODEL_PATH)
    cooccurrence_graph = torch.load("./objects/graphs/cooccurrence_graph.pth", weights_only=False)
    word_to_idx = torch.load("./objects/dictionaries/word_to_idx.pth", weights_only=False)
    idx_to_word = torch.load("./objects/dictionaries/idx_to_word.pth", weights_only=False)
    model = torch.load(MODEL_PATH, weights_only=False)
    prediction = predict_embeddings(sentences, 
                                    cooccurrence_graph, 
                                    word_to_idx, 
                                    idx_to_word, 
                                    model, 
                                    nlp, 
                                    model_fasttext)
    bank_embeddings = [i["bank"] for i in prediction[0]]
    bank_embeddings = torch.stack(bank_embeddings).numpy()
    df = pd.DataFrame(bank_embeddings)
    df['label'] = labels
    df.to_csv(os.path.join(root_path,
                             "data", 
                             "predict", 
                             "bank_sentences", 
                             "bank_embeddings.csv"), index=False)

if __name__ == "__main__":
    main()