import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.cluster import KMeans
import seaborn as sn

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
print("Nombre de classes disponible :", len(set(etiquettes)), set(etiquettes))


nbElementClasse = []
listeLabels = list(etiquettes.values) # récupération des étiquettes sous forme de liste

# Pour chaque classes comptage du nombre d'élement
for classe in set(etiquettes) :
  nbElementClasse.append(listeLabels.count(classe))

print(nbElementClasse)

# Affichage de la repartition sous forme de graphe
dfCount = pd.DataFrame([nbElementClasse])
graphe = sn.barplot(data = dfCount)
graphe.set(xlabel = "classes", ylabel = "échantillons")


#_______________Séparation des données_____________
# Séparation aléatoire du jeux de données en 2 échantillons : train (70%) / test (30%)
feature_train, feature_test, label_train, label_test = train_test_split (caracteristiques, etiquettes, test_size = 0.30, random_state = 42)

# Faire graphe data repartition
# Faire gread search
# Créer les modèles
