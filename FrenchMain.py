#_______________IMPORTS_____________
# Chargement du jeux de données
import pandas as pd

# Préparation / Manipulation / Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Graphes
import seaborn as sn
import matplotlib.pyplot as plt

# Models
from sklearn import svm, tree
from sklearn.cluster import KMeans


#_______________CHARGEMENT_____________
# Chargement du fichier csv sous forme de Dataframe
df = pd.read_csv("Pima_Indians_Diabetes.csv")
print("Colonnes : ", list(df.columns))
print("Échantillons \n", df.head)
print("Taille des données", len(df))


#_______________Nettoyage_____________
# Suppression des données vides ne pouvant être ajouter
df.dropna(inplace=True)
print("Taille des données", len(df))


#_______________Repartition_____________
# Séparation des caractéristiques et des étiquettes
# Généralement les caractéristiques / features sont appelé (x) et les étiquettes / labels sont appelé (y)
features = df.copy()
del features["Outcome"]

labels = df["Outcome"]

# Comptage du nombre de classe
nbClasse = len(set(labels))
print("Nombre de classes disponible :", nbClasse, set(labels))

# récupération des étiquettes sous forme de liste
listeLabels = list(labels.values) 
nbElementClasse = []

# Pour chaque classes comptage du nombre d'élément
for classe in set(labels) :
  nbElementClasse.append(listeLabels.count(classe))


# Affichage de la repartition sous forme de graphe
dfCount = pd.DataFrame([nbElementClasse])
graphe = sn.barplot(data = dfCount)
graphe.set(xlabel = "classes", ylabel = "échantillons")
plt.figure(figsize=(8,4))


#_______________Séparation des données_____________
# Séparation aléatoire du jeux de données en 2 échantillons : train (70%) / test (30%)
feature_train, feature_test, label_train, label_test = train_test_split (features, labels, test_size = 0.30, random_state = 42)

# création de 2 listes permettant d'associer chaque étiquettes au lot correspondant (train / test) 
trainIndicator = ["Train" for element in label_train]
testIndicator = ["Test" for element in label_test]
datasetIndicator = trainIndicator + testIndicator

datasetLabel = list(label_train.values) + list(label_test.values)

# Graphe de repartition de l'ensemble de données après séparation train / test
dfRepartition = pd.DataFrame(list(zip(datasetIndicator, datasetLabel)), columns =['labels', 'set'])
sn.countplot(data = dfRepartition, x = 'labels', hue = 'set')


# ____________________Creations des modeles____________
modelListe = {}

# SVM
modelListe["svm"] = svm.SVC()

# Arbre décision
modelListe["arbre"] = tree.DecisionTreeClassifier()

# Kmeans
modelListe["kmeans"] = KMeans(n_clusters=nbClasse)


# ____________________Selectionner uniquement les caracteristiques pertinantes____________
# Faire gread search


#____________________Entrainement____________
# On fit chaque modèle en donnant en paramètre le caractéristique du jeux d'entrainement + les etiquettes
for model in modelListe.keys() :
  print("Train", model)
  modelListe[model].fit(feature_train, label_train)

# ____________________Evaluation____________

performances = {}

# Pour chaque modèle de la liste
for model in modelListe.keys() :

  # Prediction de l'ensemble de test
  predictions = modelListe[model].predict(feature_test)

  # Evaluation du modeles en comparant les labels predit et les originaux
  perfModel = {"accuracy" : 0.0, "precision" : 0.0, "recall" : 0.0, "f1-score" : 0.0}

  # Calcule de Performance : Combien de données sont prédit correctement
  acc = accuracy_score(predictions, label_test)

  # Calcule de précision : Nombre correctement predit pour une classe / total des predictions faite pour cette classe
  precision = precision_score(predictions, label_test)

  # Calcule du rappel : Nombre d'element correctement prédit  / total à prédire
  recall = recall_score(predictions, label_test)

  # Relation entre Précision et Rappel
  f1Score = f1_score(predictions, label_test)

  perfModel["accuracy"] = acc
  perfModel["precision"] = precision
  perfModel["recall"] = recall
  perfModel["f1-score"] = f1Score

  performances[model] = perfModel

# Affichage des performances pour chaque modèle
for model in performances.keys():
  print(model, performances[model])
