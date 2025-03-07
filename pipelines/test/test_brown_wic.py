import os
import sys
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import torch
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)

from datasets import load_dataset
from pipelines.train.brown.training_brown import ContextRGAT
from pipelines.test.predict import predict_embeddings
from utils.helper_functions import load_stanza_pipeline, load_fasttext_model
import re

def process_dataset(cooccurrence_graph,
                    word_to_idx, idx_to_word,
                    model, nlp, model_fasttext,
                    dataset_split, split_name):
    predictions_list = []
    for i in tqdm(dataset_split):
        word = i['word']
        sentence_1 = (re.sub(r"[^\w\s]", "", i['sentence1']).lower()).split()
        sentence_2 = (re.sub(r"[^\w\s]", "", i['sentence2']).lower()).split()
        label = i['label']
        if (word in word_to_idx) and (word in sentence_1) and (word in sentence_2) and (len(sentence_1) > 3) and (len(sentence_2) > 3):
            try:
                sentences = [sentence_1, sentence_2]
                predictions = predict_embeddings(sentences, cooccurrence_graph, word_to_idx, idx_to_word, model, nlp, model_fasttext)
                cosine_similarity = F.cosine_similarity(predictions[0][0][word].unsqueeze(0), 
                                                        predictions[0][1][word].unsqueeze(0))
                predictions_list.append([word, sentence_1, sentence_2, cosine_similarity.item(), label])
            except Exception as e:
                continue
    # Save predictions to CSV
    pd.DataFrame(predictions_list, columns=["word", "sentence1", "sentence2", "cosine_similarity", "label"]).to_csv(
        os.path.join(root_path, "data", "predict", f"wic_{split_name}", f"wic_{split_name}.csv"), index=False)

def main():
    # Load WiC dataset
    dataset = load_dataset("super_glue", "wic")

    # Load models and data
    STANZA_DIR = os.path.expanduser("~/stanza_resources/")
    FASTTEXT_MODEL_PATH = './utils/fasttext/cc.en.300.bin'
    MODEL_PATH = './objects/models/brown/brown_model.pth'
    nlp = load_stanza_pipeline(STANZA_DIR)
    model_fasttext = load_fasttext_model(FASTTEXT_MODEL_PATH)
    cooccurrence_graph = torch.load("./objects/graphs/cooccurrence_graph.pth", weights_only=False)
    word_to_idx = torch.load("./objects/dictionaries/word_to_idx.pth", weights_only=False)
    idx_to_word = torch.load("./objects/dictionaries/idx_to_word.pth", weights_only=False)
    model = torch.load(MODEL_PATH, weights_only=False)

    process_dataset(cooccurrence_graph, 
                    word_to_idx,
                    idx_to_word,
                    model,
                    nlp,
                    model_fasttext,
                    dataset['train'], 'train')
    process_dataset(cooccurrence_graph, 
                    word_to_idx,
                    idx_to_word,
                    model,
                    nlp,
                    model_fasttext,
                    dataset['validation'], 'validation')

if __name__ == "__main__":
    main()