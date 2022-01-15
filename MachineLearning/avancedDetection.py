#_______________IMPORTS_____________
# Load dataset
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Kfold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

# Mask warning message to keep clean terminal result
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

#_______________LOAD_DATA_____________
# Load dataset to DataFrame
df = pd.read_csv("../Datasets/Pima_Indians_Diabetes.csv")
print("Columns : ", list(df.columns), "\n")
print(df.info(), "\n")
print("Example :\n", df.head, "\n")
print("Total data lenght :", len(df))


#_______________CLEANING_____________
# Remove all empty cells by removing the row
df.dropna(inplace=True)
print("Data lenght after cleaning :", len(df), "\n")


#_______________DATA_REPARTITION_____________
# Features and labels separation (Generaly features are call X and labels Y)
features = df.copy(deep=True)
del features["Outcome"] # delete labels collumns of feature variables

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

#_______________DATA_SEPARATION_____________
# Split the dataset into 2 sets : train (80%) / test (20%)
# ============================== TODO INCLUDE VALIDATION SET
featureTrain, featureTest, labelTrain, labelTest = train_test_split (features, labels, test_size = 0.2, random_state = 42)

# Creation of 2 lists to join each label with his set (Train / Test)
trainIndicator = ["Train" for element in labelTrain]
testIndicator = ["Test" for element in labelTest]
datasetIndicator = trainIndicator + testIndicator

datasetLabel = list(labelTrain.values) + list(labelTest.values)

# Print data repartition graph after train / test split
dfRepartition = pd.DataFrame(list(zip(datasetIndicator, datasetLabel)), columns =['labels', 'set'])
sn.countplot(data = dfRepartition, x = 'labels', hue = 'set')


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
parameterTree = {'bootstrap': [True],
				'max_depth': [3, 4, 5, 7, None],
				'max_features': ['auto', 'sqrt'],
				'min_samples_leaf': [1, 2, 4],
				'min_samples_split': [3, 5,],
				'n_estimators': [200, 400, 600, 800]}
# TODO describes parameters of mlogistic regression
parameterLR = {"C":np.logspace(-3,3,20),
				"penalty":["l2"]}

# TODO describes parameters of gradient boosting
parameterGB  = {"learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
				"min_samples_split": np.linspace(0.1, 0.5, 12),
				"min_samples_leaf": np.linspace(0.1, 0.5, 12),
				"max_depth":[3,4,5,7],
				"max_features":["log2","sqrt"],
				"criterion": ["friedman_mse",  "mae"],
				"subsample":[0.5, 0.65, 0.8, 0.85, 0.9, 1.0],
				"n_estimators":[300, 500]
    }

# TODO describes parameters of Kmeans

# ____________________MODELES_CREATION____________
modelListe = {}
# SVM
modelListe["SVM"] = GridSearchCV(SVC(), parameterSVM, scoring='accuracy', verbose=0)
# Arbre décision
modelListe["DECISION_TREE"] = GridSearchCV(DecisionTreeClassifier(),parameterTree, scoring='accuracy', verbose=0)
# Kmeans
modelListe["KMEANS"] = KMeans(n_clusters=nbClasse)

#
models = []
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))

#____________________TRAINING____________
# Fit all model of the list with the train features and train labels
modelList = list(modelListe.keys())

for model in modelListe.keys():
    curve = modelListe[model].fit(featureTrain, labelTrain)

	# TODO: Plot all learning curve

	# If model use gridSearch
    if type(modelListe[model]) ==  type(GridSearchCV(svm.SVC(), parameterSVM)) :
	    print(model, "best params:", modelListe[model].best_params_, "\n")


# ____________________EVALUATION____________

performances = {}

# For all model of the list
for model in modelList:

	# Prediction of all test elements
	predictions = modelListe[model].predict(featureTest)

	# Evaluation of the model by comparing prediction and real labels
	perfModel = {"accuracy" : 0.0, "precision" : 0.0, "recall" : 0.0, "f1-score" : 0.0}

	# Calculate accuracy : How many prediction are good ?
	acc = accuracy_score(predictions, labelTest)

	# Calculate precision : Number of correct prediction for this class / total of predictions for this class
	precision = precision_score(predictions, labelTest)

	# Calculate recall : Number of correct prediction  / total element of this class
	recall = recall_score(predictions, labelTest)

	# Relation beetwen precision and recall
	f1Score = f1_score(predictions, labelTest)

	perfModel["accuracy"] = acc
	perfModel["precision"] = precision
	perfModel["recall"] = recall
	perfModel["f1-score"] = f1Score

	performances[model] = perfModel

# Show performance of all models
accuracyList = []

for model in performances.keys():
	accuracyList.append(performances[model]["accuracy"])
	print(model, performances[model])


# TODO K-Fold Cross-Validation
names = []
scores = []
for name, model in models:
    
    kfold = cross_validate.KFold(n_splits=10, random_state=123) 
    score = cross_val_score(model, featureTrain, labelTrain, cv=kfold, scoring='accuracy').mean()
	# get metrics
    names.append(name)
    scores.append(score)
# save in dataframe
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print('\n\n CROSSVAL_METRICS',kf_cross_val,'\n \n')

# TODO plot confusion Matrix

# Print the best model
maxAcc = max(accuracyList)
print("\nTHE BEST MODEL IS :", modelList[accuracyList.index(maxAcc)],"\nWITH AN ACCURACY OF :", maxAcc*100, "%") # TODO

# Plot all model
# TODO plot decision tree
# TODO plot kmeans
# TODO plot SVM