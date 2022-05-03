#_______________IMPORTS_____________
# Load dataset
import pandas as pd

# Preparation / Manipulation / Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Save
import  pickle

#_______________LOAD_DATA_____________
# Load dataset to DataFrame
github_path = "/workspaces/DiabeteDetection/Datasets/Pima_Indians_Diabetes.csv"
local_path = "../Datasets/Pima_Indians_Diabetes.csv"

df = pd.read_csv(github_path)
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
print("Available features: ", features.shape[1])
labels = df["Outcome"]

# Count of classe in target variable
nbClasse = len(set(labels))
print("Available values :", nbClasse, "->", set(labels))

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

labelTrainList = list(labelTrain.values)
labelTestList = list(labelTest.values)

print("Data Repartition after split :")

for i in range(0,2): # For Train / Test

  infoDico = {}

  for classe in set(labels): # For each class

    split = "train" if i==0 else "test"
    labelList = labelTrainList if i==0 else labelTestList

    infoClasse = {"samples" : 0, "pourcent" : 0}
    infoClasse["samples"] = labelList.count(classe)
    infoClasse["pourcent"] = infoClasse["samples"] / len(labelList)
    infoDico[classe] = infoClasse

  
  print(split+":", infoDico)


# ____________________MODELES_CREATION____________
# let's use basics models withouts any parameters befor moving to advanced machine learnin in order to tune them.
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression(max_iter=500)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GB', GradientBoostingClassifier()))

#____________________TRAINING____________
# Fit all model of the list with the train features and train labels

for model_name, model in models :
    model.fit(featureTrain, labelTrain)

# ____________________EVALUATION____________

performances = {}

# For all model of the list
for model_name, model in models :

    # Prediction of all test elements
    predictions = model.predict(featureTest)

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

    perfModel["accuracy"] = round(acc, 3)
    perfModel["precision"] = round(precision, 3)
    perfModel["recall"] = round(recall, 3)
    perfModel["f1-score"] = round(f1Score, 3)

    performances[model_name] = perfModel

# Show performance of all models
print("Models performances :\n")

accuracyList = []

for model in performances.keys():
    accuracyList.append(performances[model]["accuracy"])
    print(model, performances[model])

# Print the best model
maxAcc = max(accuracyList)
print("\nTHE BEST ACCURACY MODEL IS :", models[accuracyList.index(maxAcc)],"\nWITH AN ACCURACY OF :", maxAcc*100, "%")

#____________________SAVE____________
with open(str(models[accuracyList.index(maxAcc)]) + ".pkl", 'wb')  as saveFile:
    pickle.dump(model[accuracyList.index(maxAcc)],  saveFile)
  
#____________________LOAD____________
# pickle.load("name_of_your_file.pkl")
