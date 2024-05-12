'''Principal Component Analysis of Epicurious Dataset.
Attempting to find patterns in data if any.'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

FILEPATH = "/Users/christophernowak/Documents/"

epi = pd.read_csv(FILEPATH + "epi_r-1.csv").dropna().reset_index(drop=True)
epi = epi.drop("title", axis = 1).dropna()
epi_scale = epi.copy()

features  = epi.columns.drop("cake")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(epi[features])
epi_scale.loc[:, features] = scaled_features

N = len(features)
model = PCA(n_components=2)
model.fit(epi_scale[features].dropna())

pcs_epi = model.transform(epi_scale[features])
epi['PC1'] = pcs_epi[:,0]
epi['PC2'] = pcs_epi[:,1]
new_features = ['PC1', 'PC2']

def vegetarian_dessert_pca_plot():
    '''plot of points labeled vegetarian and dessert'''
    
    hue = epi[['vegetarian', 'dessert']].apply(lambda row: f"{row.vegetarian}, {row.dessert}", axis=1)
    hue.name = 'vegetarian, dessert'
    plt.subplots(figsize=(10, 6))
    sns.scatterplot(x = epi['PC1'], y = epi['PC2'], hue = hue)
    plt.show()
    
def lunch_dinner_pca_plot():
    '''plot of points labeled lunch and dinner'''
    
    hue = epi[['lunch', 'dinner']].apply(lambda row: f"{row.lunch}, {row.dinner}", axis=1)
    hue.name = 'lunch, dinner'
    
    plt.subplots(figsize=(10, 6))
    sns.scatterplot(x = epi['PC1'], y = epi['PC2'], hue = hue)
    plt.show()
    
def alcoholic_drink_pca_plot():
    '''plot of points labeled alcoholic and drink'''
    
    hue = epi[['alcoholic', 'drink']].apply(lambda row: f"{row.alcoholic}, {row.drink}", axis=1)
    hue.name = 'alcoholic, drink'
    
    sns.scatterplot(x = epi['PC1'], y = epi['PC2'], hue = hue)
    plt.show()
    
def cake_pca_plot():
    '''plot of points labeled cake'''
    
    hue = epi["cake"]

    sns.scatterplot(x = epi['PC1'], y = epi['PC2'], hue = hue)
    plt.show()
    
def pca_var_explained_plot():
    '''Ploting the number of principal components
    vs the variance explained cummulatively and 
    individually'''
    
    model1 = PCA(n_components=N)
    model1.fit(epi_scale[features])
    
    plt.figure(figsize=(4,4))
    sns.scatterplot(x=range(1,N+1), 
                    y=model1.explained_variance_ratio_, 
                    s=100, alpha=0.6)
    sns.scatterplot(x=range(1,N+1), 
                    y=model1.explained_variance_ratio_.cumsum(), 
                    s=100, alpha=0.6)
    
    plt.xlabel('Principal Component')
    plt.ylabel('% variance explained')
    plt.legend(['PCs', 'Cumsum()'], bbox_to_anchor=[0, 0])
  
vegetarian_dessert_pca_plot()
lunch_dinner_pca_plot()
alcoholic_drink_pca_plot()
cake_pca_plot()
pca_var_explained_plot()