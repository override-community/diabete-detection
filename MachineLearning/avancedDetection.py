#_______________IMPORTS_____________
# Load dataset
from cProfile import label
import pandas as pd
import numpy as np

# Preparation / Manipulation / Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Graphs
import seaborn as sn
import matplotlib.pyplot as plt

# Models

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Kfold
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import cross_val_score

# Mask warning message to keep clean terminal result
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

#_______________LOAD_DATA_____________
# Load dataset to DataFrame
github_path = "/workspaces/DiabeteDetection/Datasets/Pima_Indians_Diabetes.csv"
local_path = "../Datasets/Pima_Indians_Diabetes.csv"
df = pd.read_csv(github_path)

print("Columns : ", list(df.columns), "\n")
print(df.info(), "\n")
print("Example :\n", df.head, "\n")
print("Total data lenght :", len(df))

# TODO Data Exploration
# Plot some X variables f(y)

#_______________CLEANING_____________
# Remove all empty cells by removing the row
df.dropna(inplace=True)
print("Data lenght after cleaning :", len(df), "\n")

#_______________DATA_REPARTITION_____________
# Features and labels separation (Generaly features are call X and labels Y)
features = df.copy(deep=True)
del features["Outcome"] # delete labels collumns of feature variables
print("Available features: ", features.shape[1])

labels = df["Outcome"]

# Count number of classes
nbClasse = len(set(labels))
print("Available classes :", nbClasse, "->", set(labels))

# count the number of element by classe
listeLabels = list(labels.values)
nbElementClasse = []

for classe in set(labels):
	nbElementClasse.append(listeLabels.count(classe))


# Show a repartition graph
dfCount = pd.DataFrame([nbElementClasse])
graphe = sn.barplot(data = dfCount)
graphe.set(xlabel = "classes", ylabel = "échantillons")
plt.figure(figsize=(8,4))


# ____________________SELECT BEST HYPER PARAMS____________

# SVM HYPER PARAMS
# C: This is a regulisation parameter
# Kernel: type of kernel : linear, poly, rbf, sigmoid, precomputed
# Max_Iter: It is the maximum number of iterations for the solver
parameterSVM = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001,],
              'kernel': ['rbf'],
			  'max_iter': [300, 500]}

# DECISION TREE HUPER PARAMS
# bootstrap: Create a large variety of trees by sampling both observations and features with replaceent
# criterion : The function to measure the quality of a split : gini or entropy
# max_depth : The maximum depth of the tree
# mini_simple_split : The minimum number of samples required to split an internal node
# mini_samples_leaf : The minimum number of samples required to be at a leaf node
# max_features: features that we want to sample in bootstrap
# n_estimators: This parameter controls the number of trees inside the classifier (inside forest/ combien d'arbre on veut dans la foret)
parameterDTree = {'bootstrap': [True],
				'max_depth': [3, 4, 5, 7, None],
				'max_features': ['auto', 'sqrt'],
				'min_samples_leaf': [1, 2, 4],
				'min_samples_split': [3, 5,],
				'n_estimators': [200, 400, 600, 800]}
# TODO - describes parameters of mlogistic regression
parameterLR = {"C":np.logspace(-3,3,20),
				"penalty":["l2"],
				"max_iter": [500, 800]}

# TODO - describes parameters of gradient boosting
parameterGB  = {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
				"min_samples_split": np.linspace(0.1, 0.5, 12),
				"min_samples_leaf": np.linspace(0.1, 0.5, 12),
				"max_depth":[3,4,5,7],
				"max_features":["log2","sqrt"],
				"criterion": ["friedman_mse",  "mae"],
				"subsample":[0.5, 0.65, 0.8, 0.85, 0.9, 1.0],
				"n_estimators":[300, 500]
    }
	
# ____________________MODELES_CREATION____________
models = []
models.append(('SVC', GridSearchCV(SVC(), parameterSVM, scoring='accuracy', verbose=0)))
models.append(('LR', GridSearchCV(LogisticRegression(), parameterLR, scoring='accuracy', verbose=0)))
models.append(('RF', GridSearchCV(RandomForestClassifier(),parameterDTree, scoring='accuracy', verbose=0)))
models.append(('GB', GridSearchCV(GradientBoostingClassifier(), parameterGB, scoring='accuracy', verbose=0)))

#____________________TRAINING____________

# TODO - K-Fold Cross-Validation
names = []
scores = []
for name, model in models:
	kfold = KFold(n_splits=5, random_state=123, shuffle=True)
	print("train", 100- (100/5), "% test", 100/5,"%")
	score = cross_val_score(model, features, labels, cv=kfold, scoring='accuracy').mean()
	# get metrics
	names.append(name)
	scores.append(score)
	# save in dataframe
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print('\n\n CROSSVAL_METRICS',kf_cross_val,'\n \n')

# TODO plot confusion Matrix

# Plot all model
# TODO plot decision tree
# TODO plot kmeans
# TODO plot SVM