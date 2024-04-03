# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 00:26:45 2024

@author: Lenovo
"""

import pandas as pd
df= pd.read_excel('/content/Firefox-FULL-Final-version.xlsx')
df.head()

df['Part1']= df["title1"] + df["description1"] + df["steps1"]+ df["actual1"]  + df["expected1"]
df['Part2']= df["title2"] + df["description2"] +  df["steps2"]+ df["actual2"] + df["expected2"]

from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load("en_core_web_sm")

def text_processing(sentence):
    """
    Lemmatize, lowercase, remove numbers and stop words

    Args:
      sentence: The sentence we want to process.

    Returns:
      A list of processed words
    """
    sentence = [token.lemma_.lower()
                for token in nlp(sentence)
                if token.is_alpha and not token.is_stop]

    return sentence


def cos_sim(sentence1_emb, sentence2_emb):
    """
    Cosine similarity between two columns of sentence embeddings

    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column

    Returns:
      The row-wise cosine similarity between the two columns.
      For instance is sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
    """
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)

from sentence_transformers import SentenceTransformer

# Load the pre-trained model
from sentence_transformers import SentenceTransformer, models

#word_embedding_model = models.Transformer('bert-based-uncased', max_seq_length=128)
#pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model = SentenceTransformer('stsb-mpnet-base-v2')

sentence1_emb = model.encode(df['Part1'], show_progress_bar=True)
sentence2_emb = model.encode(df['Part2'], show_progress_bar=True)

# Cosine Similarity
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
df['FulBR_cosine_score'] = cos_sim(sentence1_emb, sentence2_emb)

#df.to_excel("Firefox.xlsx")