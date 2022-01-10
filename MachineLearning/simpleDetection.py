#_______________IMPORTS_____________
# Load dataset
import pandas as pd

# Preparation / Manipulation / Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn import svm, tree
from sklearn.cluster import KMeans

# Save
import  pickle

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
del features["Outcome"] # delete labels columns of feature variables

labels = df["Outcome"]

# Count number of classes
nbClasse = len(set(labels))
print("Available classes :", nbClasse, "->", set(labels))


# count the number of element by classe
dataRepartition = {}
listeLabels = list(labels.values)

for classe in set(labels):
  infoClasse = {"pourcent" : 0, "samples" : 0}
  infoClasse["samples"] = listeLabels.count(classe)
  infoClasse["pourcent"] = infoClasse["samples"] / len(labels)
  dataRepartition[str(classe)] = infoClasse

print("Data Repartition :", dataRepartition, "\n")


#_______________DATA_SEPARATION_____________
# Split the dataset into 2 sets : train (80%) / test (20%)
featureTrain, featureTest, labelTrain, labelTest = train_test_split (features, labels, test_size = 0.20, random_state = 42)

trainRepartition = {}
testRepartition = {}

labelTrainList = list(labelTrain.values)
labelTestList = list(labelTest.values)

for classe in set(labels):
  # Train
  infoClasse = {"samples" : 0, "pourcent" : 0}
  infoClasse["samples"] = labelTrainList.count(classe)
  infoClasse["pourcent"] = infoClasse["samples"] / len(labelTrainList)
  trainRepartition[classe] = infoClasse

  # Test
  infoClasse = {"samples" : 0, "pourcent" : 0}
  infoClasse["samples"] = labelTestList.count(classe)
  infoClasse["pourcent"] = infoClasse["samples"] / len(labelTestList)
  testRepartition[classe] = infoClasse

print("Data Repartition after split :\ntrain :", trainRepartition, "\ntest :", testRepartition, "\n")


# ____________________MODELES_CREATION____________
modelListe = {}

# SVM
modelListe["SVM"] = svm.SVC()

# Arbre décision
modelListe["DECISION_TREE"] = tree.DecisionTreeClassifier()

# Kmeans
modelListe["KMEANS"] = KMeans(n_clusters=nbClasse)


#____________________TRAINING____________
# Fit all model of the list with the train features and train labels
modelList = list(modelListe.keys())

for model in modelList :
  modelListe[model].fit(featureTrain, labelTrain)

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
print("\nTHE BEST ACCURACY MODEL IS :", modelList[accuracyList.index(maxAcc)],"\nWITH AN ACCURACY OF :", maxAcc*100, "%")

#____________________SAVE____________
with open(modelList[accuracyList.index(maxAcc)] + ".pkl", 'wb')  as saveFile:
  pickle.dump(modelListe[modelList[accuracyList.index(maxAcc)]],  saveFile)
  
#____________________LOAD____________
# pickle.load("name_of_your_file.pkl")
