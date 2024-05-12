'''Logistic Regression and Regularization of the
principal components of the Epicurious dataset'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

FILEPATH = "/Users/christophernowak/Documents/"

epi = pd.read_csv(FILEPATH + "epi_r-1.csv").dropna().reset_index(drop=True)
epi = epi.drop("title", axis = 1).dropna()
epi_scale = epi.copy()
features  = epi.columns.drop("cake")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(epi[features])
epi_scale.loc[:, features] = scaled_features

N = len(features)
model1 = PCA(n_components=N)
model1.fit(epi_scale[features])

pcs_epi1 = model1.transform(epi[features])
pc_epi_df = pd.DataFrame({})
for i in range(len(pcs_epi1[0])):
    pc_epi_df["PC" + str(i+1)] = pcs_epi1[:,i]
pc_epi_df["cake"] = epi["cake"]

features  = pc_epi_df.columns.drop("cake")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(pc_epi_df[features])
pc_epi_df.loc[:, features] = scaled_features

pc_epi_df_slice = pc_epi_df.sample(420, random_state = 37)

X = pc_epi_df_slice.drop("cake", axis = 1)
y = pc_epi_df_slice["cake"]
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 37)

c_vals_d = {'C': np.logspace(-5, 5, 50)}
model2 = lr(penalty = "l1", solver = "liblinear")
c_find = GridSearchCV(model2, c_vals_d, cv=5, scoring='roc_auc')
c_find.fit(X_train, y_train)

opt_model = c_find.best_estimator_
test_probs = opt_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probs)

def best_c():
    '''return the best l1 regularization parameter'''
    
    return "Optimal value of C: " + str(np.round(opt_model.C))

def best_auc():
    '''return the best auc score under the best 
    l1 regularization parameter'''
    
    return "AUC of the test set at C = 222: " + str(test_auc)

def l1_regularization_plot():
    '''plot l1 regularization parameter against
    feature coefficent importance'''
    
    c_vals = np.logspace(-5, 5, 50)

    feature_coefs = []
    log_c_vals = []
    
    for c in c_vals:
        model3 = lr(penalty = "l1", solver = "liblinear", C = c)
        model3.fit (X_train, y_train)
        
        log_c_vals.append(np.log(c))
        feature_coefs.append(model3.coef_.flatten())
        
    log_c_vals = np.array(log_c_vals)
    feature_coefs = np.array(feature_coefs)
    
    plt.figure(figsize=(10, 6))
    for i in range(feature_coefs.shape[1]):
        plt.plot(log_c_vals, feature_coefs[:, i], label="PC"+str(i+1))
    
    plt.xlabel('log(C)')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Coefficients vs. log(C) for L1 Regularization')
    plt.show()

l1_regularization_plot()
print(best_auc())