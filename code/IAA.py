# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:38:44 2025

@author: KeliDu
"""

from statsmodels.stats import inter_rater as irr
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt

os.chdir(r'DHd2026\annotation_all')

def fleiss_kappa(df): 
    dats, cats = irr.aggregate_raters(df)
    fleiss_kappa = irr.fleiss_kappa(dats, method='randolph')
    return fleiss_kappa

genres = ['scifi', 'policier', 'sentimental', 'blanche']
measures = ['LLR', 'Zeta', 'WelchT']

all_results = []

for genre in genres:
    for measure in measures:
        df = pd.read_csv(genre + '_' + measure + '.csv', sep='\t')
        df_interpretable = df[['interpretable_JR','interpretable_ER','interpretable_MR']]
        IAA_interpretable = fleiss_kappa(df_interpretable)
        all_results.append((genre, measure, IAA_interpretable))

IAA_all = pd.DataFrame(all_results, columns=['genre', 'measure', 'IAA_interpretable'])

visual = IAA_all.melt(id_vars=['measure', 'genre'], value_vars=['IAA_interpretable'], var_name='annotation', value_name='Fleiss_kappa')
#visual = IAA_all.melt(id_vars=['topic_model', 'genre', 'comparison'], value_vars=['IAA_interpretable?', 'IAA_category'], var_name='annotation', value_name='Fleiss_kappa')
visual.to_csv(r'IAA_results.csv', sep='\t', index=False)
visual = visual.sort_values('genre')

g = sns.FacetGrid(visual, col="genre", col_wrap=2, height=4, aspect=1.5)
g.map_dataframe(sns.barplot, x="annotation", y="Fleiss_kappa", hue="measure", palette='colorblind')#, dodge=True).set(yscale = 'log')
g.add_legend()
plt.show()


df = pd.read_csv(r'interpretable_count.csv', sep='\t', index_col=0)

sns.set(font_scale=2.5)
f, ax = plt.subplots(figsize = (20,15))
g = sns.heatmap(df, annot=True, cmap='rocket_r', vmin=3, vmax=10, annot_kws={'size': 36})
plt.show()

