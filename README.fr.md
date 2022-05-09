

![License: MIT](https://img.shields.io/badge/Licence-MIT-green)
![dependencies: latest](https://img.shields.io/badge/dependencies-latest-brightgreen)
![deep: learning](https://img.shields.io/badge/deep-learning-blue)
![machine: learning](https://img.shields.io/badge/machine-learning-blue)
![python: 3.7](https://img.shields.io/badge/python-3.7-blue)
![Contributors](https://img.shields.io/badge/contributor-2-orange)
![Stars](https://img.shields.io/github/stars/override-community/diabete-detection?color=orange)
![Fork](https://img.shields.io/github/forks/override-community/diabete-detection?color=orange)
![Watchers](https://img.shields.io/github/watchers/override-community/diabete-detection?color=orange)

<!DOCTYPE html>

<html>
<h1 align="center"> Introduction à l'apprentissage automatique : détection du diabète </h1>
<h3 align="center"> LANGUAGES : <a href ="https://github.com/override-community/diabete-detection/blob/main/README.md"> ENGLISH</a> / <a href ="https://github.com/override-community/diabete-detection/blob/main/README.es.md"> ESPAÑOL</a> </h3>

<h2><u> Contexte : </u></h2>
Niveau : <image src="Ressource/easy_lvl.png" width=100>

Ce projet a pour but de vous faire découvrir le domaine du machine learning à travers un exercice simple : la détection automatique du diabète.
Il est donc dédié aux débutants et à toute personne souhaitant découvrir ce domaine.
Étant donné que ce projet fait office d'introduction, vous ne trouvez ici que des concepts simples ainsi que le strict minimum.
Donc si jamais vous êtes déjà expérimenté dans le domaine, n'hésitez pas à jeter un œil à nos autres projets.


<h2><u>Architecture du projet :</h2></u>
Ce projet est composé de 3 dossiers et d'un article Medium <i>(dispo ici)</i> contenant notre démarche ainsi que des explications supplémentaires sur le domaine. <br><br>

<ul>
<li> Le dossier dataset: <br>
  contient le jeu de données <a href ="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database">"Pima_Indians_Diabetes"</a> provenant de kaggle</li>

<li> Le dossier machine learning : <br>
  contenant un code traitant la problématique d'un point de vue machine learning ainsi qu'un fichier contenant les installations requises. </li>

<li> Le dossier deep learning : <br>
  contient aussi un fichier d'installation ainsi qu'un code traitant la problématique d'un point de vue deep. </li>

</ul>

<h2><u> Jeu de données / Dataset : </h2></u>
  Le jeu de données est un fichier CSV contenant 769 lignes.<br>
  Chacune de ces lignes correspond à un patient atteint ou non du diabète.<br><br>
  La colonne "Outome" correspond aux étiquettes / labels, elle indique donc si le patient à ou non le diabète.
  Si c'est le cas, la valeur sera de 1 sinon de 0.<br><br>
  Les 8 autres colonnes : [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age] font office de caractéristiques aidant à établir une prédiction.


  
  
<h2><u> Installation / Exécution : </h2></u>
  <h3> Installation </h3>
    Choisissez votre méthode de résolution "Machine learning" ou "Deep learning" et installer les dépendances nécessaire : </br>

    cd MachineLearning 
    pip install -r requirement.txt 

    cd DeepLearning 
    pip install -r requirement.txt
   
  <h3> Exécution </h3>
    Après installation éditer le fichier SimpleDetection et changer le chemin vers le jeu données, puis exécuter le code python. <br>
    
    # Éditez Ligne 24
    df = pd.read_csv(github_path)

    # Exécuter le code
    python simpleDetection.py


  <h2><u> Sources : </h2></u>
Jeux de données : <a href ="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"> https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database </a> <br>
Kaggle : <a href ="https://www.kaggle.com"> https://www.kaggle.com </a>
  
<h2><u> Réseaux : </h2></u>
<p> <image src="Ressource/discord_icon.png" width=25 height=25> <a href="https://discord.gg/pgEUk9xVKe"> Discord @verride </a> </p>
<p> <image src="Ressource/medium_icon.png" width=25 height=25> <a href ="https://medium.com/@overridecommunuty" > @overridecommunuty </a> </p>
<p> <image src="Ressource/youtube_icon.png" width=25 height=25> <a href ="https://www.youtube.com/channel/UCHS2xgITwh7olsnznmq8o0A"> Youtube Fab.16 chanel </a> </p>
</html>
