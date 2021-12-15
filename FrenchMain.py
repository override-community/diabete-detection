#_______________IMPORTS_____________
# Chargement du jeux de données
import pandas as pd

# Préparation / Manipulation / Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV

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
caracteristiques = df.copy()
del caracteristiques["Outcome"]

etiquettes = df["Outcome"]

# Comptage du nombre de classe
nbClasse = len(set(etiquettes))
print("Nombre de classes disponible :", nbClasse, set(etiquettes))


# récupération des étiquettes sous forme de liste
listeLabels = list(etiquettes.values) 
nbElementClasse = []

# Pour chaque classes comptage du nombre d'élément
for classe in set(etiquettes) :
  nbElementClasse.append(listeLabels.count(classe))

# Affichage de la repartition sous forme de graphe
dfCount = pd.DataFrame([nbElementClasse])
graphe = sn.barplot(data = dfCount)
graphe.set(xlabel = "classes", ylabel = "échantillons")
plt.figure(figsize=(8,4))

#_______________Séparation des données_____________
# Séparation aléatoire du jeux de données en 2 échantillons : train (70%) / test (30%)
feature_train, feature_test, label_train, label_test = train_test_split (caracteristiques, etiquettes, test_size = 0.30, random_state = 42)

# création de 2 listes permettant d'associer chaque étiquettes au lot correspondant (train / test) 
lotTrain = ["Train" for element in label_train]
lotTest = ["Test" for element in label_test]
datasetLot = lotTrain + lotTest

datasetLabel = list(label_train.values) + list(label_test.values)

# Graphe de repartition de l'ensemble de données après séparation train / test
dfRepartition = pd.DataFrame(list(zip(datasetLabel, datasetLot)), columns =['etiquettes', 'lot'])
sn.countplot(data = dfRepartition, x = 'etiquettes', hue = 'lot')

# ____________________Creations des modeles____________
svm = svm.SVC()
arbre = tree.DecisionTreeClassifier()
kmeans = KMeans(n_clusters=nbClasse)

# ____________________Selectionner uniquement les caracteristiques pertinantes____________
# Faire gread search
