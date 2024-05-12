'''K-Means clustering of principal components'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator

FILEPATH = "/Users/christophernowak/Documents/"

epi = pd.read_csv(FILEPATH + "epi_r-1.csv").dropna().reset_index(drop=True)
epi = epi.drop("title", axis = 1).dropna()
epi_scale = epi.copy()
epi_kmeans = epi.copy()
features  = epi.columns.drop("cake")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(epi_kmeans)
epi_kmeans.loc[:] = scaled_features
epi_kmeans

def elbow_locator_plot():
    n_lst = range(1,13)
    inertia_lst = []

    for n in n_lst:
        model4 = KMeans(n_clusters=n)
        model4.fit(epi_kmeans)
        inertia_lst.append(model4.inertia_)
        
    kl = KneeLocator(n_lst, inertia_lst, curve='convex', direction='decreasing')
    print('Automatically found elbow at %d clusters.'%kl.elbow)

    plt.figure(figsize = (10,6))
    plt.plot(n_lst, inertia_lst)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Variance (Inertia)')
    plt.title('Elbow plot for KMeans')
    plt.show()
    
    return kl.elbow
    
model5 = PCA(n_components=len(epi_kmeans.columns))
model5.fit(epi_kmeans)

pcs_epi2 = model5.transform(epi_kmeans)
pc_epi_df2 = pd.DataFrame({})
c = 0
for i in range(len(pcs_epi2[0])):
    pc_epi_df2["PC" + str(i+1)] = pcs_epi2[:,i]

scaler1 = StandardScaler()
scaled_features1 = scaler1.fit_transform(pc_epi_df2)
pc_epi_df2.loc[:, pc_epi_df2.columns] = scaled_features1

def pca_clustering_plots(n):
    n_clusters = n
    pairs = [(1,2), (3,4), (5,6), (7,8), (9,10)]
    for pair in pairs:
        kmeans = KMeans(init='random', n_clusters=n_clusters,
                        n_init=10, max_iter=300)
        features = ['PC'+str(pair[0]), 'PC'+str(pair[1])]
        kmeans.fit(pc_epi_df2[features])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.scatterplot(ax=ax, 
                        x=pc_epi_df2['PC'+str(pair[0])], 
                        y=pc_epi_df2['PC'+str(pair[1])],
                        hue=kmeans.labels_, 
                        palette=sns.color_palette('tab10', 
                                                  n_colors=n_clusters),
                        legend=None, alpha = 0.33)
    
        for n, [pc1, pc2] in enumerate(kmeans.cluster_centers_):
            ax.scatter(pc1, pc2, s=80, marker='x', c='#a8323e');
    
        plt.xlabel("Variance explained by PC"+str(pair[0])+": " 
                   + str(model5.explained_variance_ratio_[pair[0]]))
        plt.ylabel('Variance explained by PC'+str(pair[1])+": " 
                   + str(np.round(model5.explained_variance_ratio_[pair[1]], 5)))
        plt.show()

elbow_locator_plot()
pca_clustering_plots(elbow_locator_plot())