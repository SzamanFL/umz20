#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import relplot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = pd.read_csv('mieszkania4.tsv', sep='\t')

data = data[
    (data['Powierzchnia w m2'] < 10000)
    & (data['cena'] < 10000000)
    ]
X_columns = [
    'cena',
    'Powierzchnia w m2',
    'Liczba pokoi', 
    'Liczba pięter w budynku', 
    'Piętro',
    ]
data['Piętro'] = data['Piętro'].apply(lambda x: 0 if x in ['parter', 'niski parter'] else x)
data['Piętro'] = data['Piętro'].apply(pd.to_numeric, errors='coerce')
data=data[:100]
X = data[X_columns].dropna().reset_index().drop(['index'], axis=1)

kmeans = KMeans(n_clusters=5).fit(X.values)
labels = pd.DataFrame(kmeans.labels_, columns=['label'])
labeled = pd.concat([labels, X], axis=1)

print(labeled)

relplot(data=labeled, x='Liczba pokoi', y='Piętro', hue='label')
relplot(data=labeled, x='Powierzchnia w m2', y='cena', hue='label')
plt.show()

pca = PCA(n_components=2)
pca.fit(X)

X_transformed = pd.DataFrame(pca.transform(X), columns=['x1', 'x2'])
labeled_transformed = pd.concat([labels, X_transformed, X], axis=1)

relplot(data=labeled_transformed, x='x1', y='x2', hue='label')
plt.show()