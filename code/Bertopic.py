# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 00:02:06 2023

@author: KeliDu
"""
from bertopic import BERTopic
import os
import pandas as pd

stopwords = open(r'stopwords.txt', 'r', encoding='utf-8').read().split('\n')

corpus_path = r'segments_500/'
filenames = sorted([os.path.join(corpus_path, fn) for fn in os.listdir(corpus_path)])
docs = []
for file in filenames:
    text = open(file, 'r', encoding='utf-8').read()
    #print(os.path.basename(file))
    docs.append(text)
print('Texts load.')

topic_num = 50

################################################################################################################################
'''Embeddings'''
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("all-MiniLM-L12-v2")
print('Embeddings load.')

################################################################################################################################
'''Dimensionality Reduction'''
from umap import UMAP
#model_1
dim_model = UMAP(n_neighbors=10, n_components=10, min_dist=0.001, metric='cosine', random_state=42)
#model_2
#dim_model = UMAP(n_neighbors=30, n_components=10, min_dist=0.001, metric='cosine', random_state=42)
#model_3
#dim_model = UMAP(n_neighbors=50, n_components=15, min_dist=0.001, metric='cosine', random_state=42)
#model_4
#dim_model = UMAP(n_neighbors=5, n_components=15, min_dist=0.1, metric='cosine', random_state=42)
print('Dimensionality reduction load.')

################################################################################################################################
'''Clustering'''
import hdbscan
cluster_model = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data = True)
#cluster_model = KMeans(n_clusters=topic_num)
print('Clustering load.')

################################################################################################################################
'''Vectorizers'''
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words=stopwords, min_df=2, ngram_range=(1, 3))
print('Vectorizers load.')

################################################################################################################################
'''c-TF-IDF'''
from bertopic.vectorizers import ClassTfidfTransformer
ctfidf_model = ClassTfidfTransformer()
print('c-TF-IDF load.')


os.environ["TOKENIZERS_PARALLELISM"] = "true"

################################################################################################################################
'''train topic model'''
topic_model = BERTopic(embedding_model=sentence_model, hdbscan_model=cluster_model, ctfidf_model=ctfidf_model, umap_model=dim_model, vectorizer_model=vectorizer_model, low_memory=False)#, calculate_probabilities=True, low_memory=False)
topics, probs = topic_model.fit_transform(docs)
#topic_model.update_topics(docs, vectorizer_model=vectorizer_model)
print('Topic model trained.')

################################################################################################################################

embedding_model = "sentence-transformers/all-MiniLM-L12-v2"
#topic_model.save("model_segment_500_3gram_100topics", serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)

with open ('topics_probs.txt', 'w', encoding='utf-8') as fout:
    for i in probs:
        fout.write(str(i) + ',')

with open ('topics.txt', 'w', encoding='utf-8') as fout:
    fout.write(str(topics))

topic_info = pd.DataFrame(topic_model.get_topic_info())
topic_info.to_csv('topic_info.csv', sep='\t', index=False)

output = []
x = 0
while x < topic_num:
    topic =  topic_model.get_topic(x)
    topic_words = []
    for line in topic:
        word = line[0]
        topic_words.append(word)
    output.append(topic)
    x+=1

with open ('topic_words.txt', 'w', encoding='utf-8') as fout:
    for i in output:
        fout.write(str(i))








