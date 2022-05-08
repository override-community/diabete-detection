<!DOCTYPE html>

<html>
<h1 align="center"> Introduction to machine learning: diabetes detection </h1>

<h3 align="center"> LANGUES : <a href ="https://github.com/override-community/diabete-detection/blob/main/README.fr.md"> FRENCH</a> / <a href ="https://github.com/override-community/diabete-detection/blob/main/README.es.md"> ESPAÑOL</a> </h3>

<h2><u> News : </u></h2>
  
<h2><u> Contexte : </u></h2>
Niveau : <image src="Ressource/easy_lvl.png" width=100>

This project aims to make you discover the field of machine learning through a simple exercise: the automatic detection of diabetes.
It is therefore dedicated to beginners and to anyone who wants to discover this field.
As this project is an introduction, you will find here only simple concepts and the bare minimum.
So if you are already experienced in the field, don't hesitate to have a look at our other projects.


<h2><u>Architecture du projet :</h2></u>
This project is composed of 3 folders and a Medium article <i>(here)</i> containing our approach and additional explanations on the field.
  
<ul>
<li> The dataset folder: <br>
  contains the data set <a href ="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database">"Pima_Indians_Diabetes"</a> from kaggle</li>

<li> The machine learning folder : <br>
  Contains a code treating the problem from a machine learning point of view as well as a file containing the required facilities. </li>

<li> The deep learning folder: <br>
  Contains a code treating the problem from a deep learning point of view as well as a file containing the required facilities.</li>

</ul>

<h2><u> Jeu de données / Dataset : </h2></u>
  The dataset is a CSV file containing 769 rows.<br>
  Each of these lines corresponds to a patient with or without diabetes.<br><br>
  The column "Outome" corresponds to the labels, so it indicates if the patient has diabetes `valued 1` or not `valued 0`.
  The 8 other features : [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age] are used as descriptionn to help make a prediction.


  
  
<h2><u> Installation / Execution : </h2></u>
  <h3> Installation </h3>
    Choose your solving method "Machine learning" or "Deep learning" and install the necessary dependencies : </br>
    
    
    cd MachineLearning 
    pip install -r requirement.txt  
  
    cd DeepLearning 
    pip install -r requirement.txt
   
  <h3> Exécution </h3>
    After installation edit the SimpleDetection file and change the path to the dataset, then run the python code. <br>
    
    # Éditez Ligne 24
    df = pd.read_csv(github_path)

    # Exécuter le code
    python simpleDetection.py


<h2><u> Sources : </h2></u>
Jeux de données : <a href ="https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"> https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database </a> <br>
Kaggle : <a href ="https://www.kaggle.com"> https://www.kaggle.com </a>
  
<h2><u> Réseaux : </h2></u>
<p> <image src="Ressource/discord_icon.png" width=25 height=25> <a href="https://discord.gg/pgEUk9xVKe"> https://discord.gg/pgEUk9xVKe </a> </p>
<p> <image src="Ressource/medium_icon.png" width=25 height=25> <a href ="" >  </a> </p>
<p> <image src="Ressource/youtube_icon.png" width=25 height=25> <a href ="https://www.youtube.com/channel/UCHS2xgITwh7olsnznmq8o0A"> https://www.youtube.com/channel/UCHS2xgITwh7olsnznmq8o0A </a> </p>
</html>
