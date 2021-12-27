#_______________IMPORTS_____________
# Load dataset
import pandas as pd

# Preparation / Manipulation / Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Graphs
import seaborn as sn
import matplotlib.pyplot as plt

# Models
from sklearn import svm, tree
from sklearn.cluster import KMeans

# Mask warning message to keep clean terminal result
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

#_______________LOAD_DATA_____________
# Load dataset to DataFrame
df = pd.read_csv("Pima_Indians_Diabetes.csv")
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
features = df.copy()
del features["Outcome"] # delete labels collumns of feature variables

labels = df["Outcome"]

# Count number of classes
nbClasse = len(set(labels))
print("Available classes :", nbClasse, "->", set(labels))

# count the number of element by classe
listeLabels = list(labels.values) 
nbElementClasse = []

for classe in set(labels) :
  nbElementClasse.append(listeLabels.count(classe))


# Show a repartition graph
dfCount = pd.DataFrame([nbElementClasse])
graphe = sn.barplot(data = dfCount)
graphe.set(xlabel = "classes", ylabel = "échantillons")
plt.figure(figsize=(8,4))

#_______________DATA_SEPARATION_____________
# Split the dataset into 2 sets : train (70%) / test (30%)
# ============================== TODO INCLUDE VALIDATION SET
featureTrain, featureTest, labelTrain, labelTest = train_test_split (features, labels, test_size = 0.30, random_state = 42)

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
parameterSVM = [{
  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
  'C': [1, 10, 50, 100, 250, 500, 1000],
  'max_iter': [10, 50, 100, 250]
  }]

# DECISION TREE HUPER PARAMS
# criterion : The function to measure the quality of a split : gini or entropy
# max_depth : The maximum depth of the tree
# mini_simple_split : The minimum number of samples required to split an internal node
# mini_samples_leaf : The minimum number of samples required to be at a leaf node
parameterTree = [{
  'criterion': ['gini', 'entropy'], 
  }]


# ____________________MODELES_CREATION____________
modelListe = {}

# SVM
modelListe["SVM"] = GridSearchCV(svm.SVC(), parameterSVM, scoring='accuracy', verbose=0)

# Arbre décision
modelListe["DECISION_TREE"] = GridSearchCV(tree.DecisionTreeClassifier(), parameterTree, scoring='accuracy', verbose=0)

# Kmeans
modelListe["KMEANS"] = KMeans(n_clusters=nbClasse)


#____________________TRAINING____________
# Fit all model of the list with the train features and train labels
modelList = list(modelListe.keys())

for model in modelListe.keys() :
  curve = modelListe[model].fit(featureTrain, labelTrain)

  # TODO Plot all learning curve

  # If model use gridSearch
  if type(modelListe[model]) ==  type(GridSearchCV(svm.SVC(), parameterSVM)) :
    print(model, "best params:", modelListe[model].best_params_, "\n")


# ____________________EVALUATION____________

performances = {}

# For all model of the list
for model in modelList :

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


# Print the best model
maxAcc = max(accuracyList)
print("\nTHE BEST MODEL IS :", modelList[accuracyList.index(maxAcc)],"\nWITH AN ACCURACY OF :", maxAcc*100, "%")

# Plot all model
#tree.plot_tree(modelListe["DECISION_TREE"])
# TODO plot decision tree
# TODO plot kmeans
# TODO plot SVM
