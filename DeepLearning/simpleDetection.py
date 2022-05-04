#_______________IMPORTS_____________
# Load dataset
import pandas as pd

# Preparation / Manipulation / Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Matrice manipulation
import numpy as np

# Models
from keras.models import Sequential
from keras.layers import Dense 

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

# delete labels columns of feature variables
features = df.copy()
del features["Outcome"]

print("Available features: ", features.shape[1])
labels = df["Outcome"]

# Count of classe in target variable
nbClasse = len(set(labels))
print("Available classes :", nbClasse, "->", set(labels))


# count the number of element by classe
dataRepartition = {}
listeLabels = list(labels.values)

for classe in set(labels):
  dataRepartition[str(classe)] = listeLabels.count(classe)

print("Data Repartition :", dataRepartition, "\n")


#_______________DATA_SEPARATION_____________
# Split the dataset into 2 sets : train (70%) / test (30%)
featureTrain, featureTest, labelTrain, labelTest = train_test_split (features, labels, test_size = 0.30, random_state = 42)

trainRepartition = {}
testRepartition = {}

labelTrainList = list(labelTrain.values)
labelTestList = list(labelTest.values)

for classe in set(labels):
  trainRepartition[classe] = labelTrainList.count(classe)
  testRepartition[classe] = labelTestList.count(classe)

print("Data Repartition after split :\ntrain :", trainRepartition, "\ntest :", testRepartition, "\n")


# ____________________MODELES_CREATION____________

featureToList = list(featureTrain.values)

# number of feature for a sample (patient)
nbFeature = len(featureToList[0]) 
print("Number of feature for a sample", nbFeature)

# Short model with one hidden layer (hidden layer = layer beetween input / output layer)
# Construct empty model
model = Sequential() 

# add input layer
# dense = vote layer
# 12 are the number of neurons
# input_dim = number of feature for a sample
model.add(Dense(12, activation = 'relu', input_dim = nbFeature)) 
          
# add vote layer          
model.add(Dense(8, activation = 'relu'))
          
# add output layer
# use sofmax activation for depending probability inter class
model.add(Dense(nbClasse, activation = 'softmax'))
model.compile(optimizer='rmsprop', loss='mse')


#____________________TRAINING____________
# Fit the model with the train features and train labels
# batch_size represente the number of sample it will see at the same time
# epochs represente the number of time he will see all data
model.fit(featureTrain,labelTrain,batch_size=32,epochs=3)

# Predict all data
prediction = model.predict(featureTest)

# Take the best classe (hight score) for all prediction
bestPred = [np.argmax(pred) for pred in prediction]

# Calculate accuracy : How many prediction are good ?
acc = accuracy_score(bestPred, labelTest)

# Calculate precision : Number of correct prediction for this class / total of predictions for this class
precision = precision_score(bestPred, labelTest)

# Calculate recall : Number of correct prediction  / total element of this class
recall = recall_score(bestPred, labelTest)

# Relation beetwen precision and recall
f1Score = f1_score(bestPred, labelTest)

print("\nAccuracy:", acc*100, "\nPrecision :", precision*100, "\nRecall", recall*100, "\nF1 score", f1Score*100)
