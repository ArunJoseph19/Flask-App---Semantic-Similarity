# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:52:41 2022

@author: A ARUN JOSEPHRAJ
"""
import pandas as pd
df = pd.read_csv('./Precily_Text_Similarity.csv')

df.head()

text_1 = df['text1'].to_list()
text_2 = df['text2'].to_list()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/nli-distilroberta-base-v2')

embeds_1 = model.encode(text_1[:10])
embeds_2 = model.encode(text_2[:10])

from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity([embeds_1[9]],embeds_2))
