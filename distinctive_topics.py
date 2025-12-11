# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:14:36 2025

@author: KeliDu
"""

import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

os.chdir(r'DHd2026')

##########################################################################################
#topic_info
topic_info = pd.read_csv(r'trained_model/topic_info.csv', sep='\t')

sns.set(font_scale=1.2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (20,10))
g = sns.pointplot(x='Topic', y='Count', data=topic_info.head(51), errorbar=None, linestyle='')
ax.set_yscale('log')
plt.show()

##########################################################################################
#topics of each segment

topic_assignment = open(r'trained_model/topics.txt', 'r', encoding='utf-8').read().split(', ')

segment_names = open(r'segment_names.txt', 'r', encoding='utf-8').read().split('\n')

metadata = pd.read_csv(r'metadata.csv', sep='\t')

subgenre = []
book = []
for name in segment_names:
    book.append(name[:-12])
    row = metadata[metadata['idno'] == name[:-12]]
    subgenre.append(row['subgenre'].item())

all_df = pd.DataFrame({
    'topic_assignment': topic_assignment,
    'segment_names': segment_names, 
    'subgenre': subgenre,
    'book': book})
all_df = all_df.astype({"topic_assignment": int})

##########################################################################################
#topics in each 5000-words chunk
book_names = sorted(list(set(book)))

all_dfs = []
for book_name in book_names:
    book_df = all_df[all_df['book'] == book_name]
    book_df['chunk']=np.divmod(np.arange(len(book_df)),10)[0]+1
    book_df['no_of_chunks'] = [book_df['chunk'].max()] * len(book_df)
    all_dfs.append(book_df[["topic_assignment", "subgenre", "book", "chunk", "no_of_chunks"]])

all_df_new = pd.concat(all_dfs)

##########################################################################################
#topic Zeta

test_genre = 'policier'

target = all_df_new[all_df_new['subgenre'] == test_genre].drop_duplicates()
comparison = all_df_new[all_df_new['subgenre'] != test_genre].drop_duplicates()
topic_freq_target = target['topic_assignment'].value_counts()
topic_freq_comparison = comparison['topic_assignment'].value_counts()
target_no_of_chunks = target[['book', 'no_of_chunks']].drop_duplicates().no_of_chunks.sum()
comparison_no_of_chunks = comparison[['book', 'no_of_chunks']].drop_duplicates().no_of_chunks.sum()

topic_no = -1
topic_zetas = []
while topic_no < len(set(all_df['topic_assignment']))-1:
    topic_words = topic_info[topic_info['Topic'] == topic_no]['Representation'].item()
    try:
        target_props = topic_freq_target[topic_no] / target_no_of_chunks
    except KeyError:
        target_props = 0
        
    try:
        comparison_props = topic_freq_comparison[topic_no] / comparison_no_of_chunks
    except KeyError:
        comparison_props = 0        
        
    topic_zeta = (target_props - comparison_props) / 2
    topic_zetas.append((topic_no, topic_zeta, topic_words))
    topic_no += 1

topic_zeta_df = pd.DataFrame(topic_zetas, columns=['topic_no', 'topic_zeta', 'topic_words'])
topic_zeta_df = topic_zeta_df.sort_values(by=['topic_zeta'], ascending=False)

topic_zeta_df.to_csv(r'distinctive_topics/' + test_genre + '_topiczeta_chunk_5000.csv', sep='\t', index = False)

##########################################################################################
#Welch's T

def get_topic_freq_per_corpus(df):
    all_topics_in_books_freqs = pd.DataFrame()
    topicNums = list(range(-1, 427))
    all_topics_in_books_freqs.index = topicNums
    books = sorted(list(set(df['book'])))
    for book in books:
        book_df = df[df['book'] == book]
        topics_in_book_freqs=book_df['topic_assignment'].value_counts()
        all_topics_in_books_freqs[book] = topics_in_book_freqs
    all_topics_in_books_freqs = all_topics_in_books_freqs.T
    all_topics_in_books_freqs = all_topics_in_books_freqs.fillna(0)
    return all_topics_in_books_freqs

def Welchs_t_test (absolute1, absolute2, topic_no):
    """
    This function implements Welch's t-test (https://en.wikipedia.org/wiki/Welch%27s_t-test)
    The input "absolute1" and "absoulte2" should be 2 dataframes. Columns represent documents and rows represents features.
    """
    welch_t_results = stats.ttest_ind(topic_freq_df_in_target, topic_freq_df_in_comparison, equal_var = False)
    welch_t_df = pd.DataFrame(welch_t_results)
    welch_t_df = welch_t_df.T
    topicNums = list(range(0, topic_no))
    welch_t_df.index = topicNums
    welch_t_df.columns = ['t_value', 'p_value']
    return welch_t_df

test_genre = 'policier'

target = all_df_new[all_df_new['subgenre'] == test_genre]
comparison = all_df_new[all_df_new['subgenre'] != test_genre]
topic_freq_df_in_target = get_topic_freq_per_corpus(target)
topic_freq_df_in_target = topic_freq_df_in_target.loc[:, topic_freq_df_in_target.columns != -1]
topic_freq_df_in_comparison = get_topic_freq_per_corpus(comparison)
topic_freq_df_in_comparison = topic_freq_df_in_comparison.loc[:, topic_freq_df_in_comparison.columns != -1]

welch_output_df = Welchs_t_test (topic_freq_df_in_target, topic_freq_df_in_comparison, 427)

welch_output_df['topic_words'] = topic_info[topic_info['Topic'] != -1].reset_index().Representation

welch_output_df = welch_output_df.sort_values(by=['t_value'], ascending=False)

welch_output_df.to_csv(r'distinctive_topics/' + test_genre + '_WelchT.csv', sep='\t', index = False)

##########################################################################################
#LLR

def LLR_test (topic_freq_sum_target, topic_freq_sum_comparison, absolute1, absolute2, topic_no):
    """
    This function implements Log-likelihood-Ratio test (https://en.wikipedia.org/wiki/G-test)
    The input "absolute1" and "absoulte2" should be 2 dataframes. Columns represent documents and rows represents features.
    """
    LLR_results = []
    LLR_count = 0
    corpus1 = topic_freq_sum_target
    corpus2 = topic_freq_sum_comparison
    absolute1_sum = absolute1.sum()
    absolute2_sum = absolute2.sum()
    while LLR_count < topic_no:
        obs1 = absolute1_sum[LLR_count]
        obs2 = absolute2_sum[LLR_count]
        exp1 = (corpus1 * (obs1 + obs2) ) / (corpus1 + corpus2)
        exp2 = (corpus2 * (obs1 + obs2) ) / (corpus1 + corpus2)
        LLR_row_result = stats.power_divergence([obs1, obs2], f_exp= [exp1, exp2], lambda_='log-likelihood')
        LLR_results.append(LLR_row_result)
        LLR_count+=1
    LLR_full = pd.DataFrame(LLR_results, columns = ['LLR_value', 'p_value'])
    return LLR_full

test_genre = 'blanche'

target = all_df_new[all_df_new['subgenre'] == test_genre]
comparison = all_df_new[all_df_new['subgenre'] != test_genre]
topic_freq_df_in_target = get_topic_freq_per_corpus(target)
topic_freq_target_sum = sum(topic_freq_df_in_target.sum())
topic_freq_df_in_target = topic_freq_df_in_target.loc[:, topic_freq_df_in_target.columns != -1]
topic_freq_df_in_comparison = get_topic_freq_per_corpus(comparison)
topic_freq_comparison_sum = sum(topic_freq_df_in_comparison.sum())
topic_freq_df_in_comparison = topic_freq_df_in_comparison.loc[:, topic_freq_df_in_comparison.columns != -1]

LLR_output_df = LLR_test (topic_freq_target_sum, topic_freq_comparison_sum, topic_freq_df_in_target, topic_freq_df_in_comparison, 427)
LLR_output_df['topic_words'] = topic_info[topic_info['Topic'] != -1].reset_index().Representation
LLR_output_df = LLR_output_df.sort_values(by=['LLR_value'], ascending=False)
LLR_output_df.to_csv(r'distinctive_topics/' + test_genre + '_LLR.csv', sep='\t', index = False)





