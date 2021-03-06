# SFM recurrent network for stock price prediction: projet Deep Learning master SID
This is the project for the following paper:
    
    Liheng Zhang, Charu Aggarwal, Guo-Jun Qi, Stock Price Prediction via Discovering Multi-Frequency Trading Patterns,
    in Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2017), Halifax, Nova Scotia,
    Canada, August 13-17, 2017.
    
 ## Pré-requis pour utiliser SFM
 Créer un env conda avec:
 - Python == 2.7
 - Keras == 1.0.1
 - Theano == 0.9
exemple d'exécution: KERAS_BACKEND=theano python sfm_papier.py

## Introduction: performance SFM sur sinus
### Dossier SFM
--> Dossier dataset: 
 - generate_data.ipynb permet de génerer des fichiers .csv des 3 sinus de fréquences respectives 10, 20 et 40
 - build_data.py permet de générer dans des formats adéquats les fichiers de données utilisés dans l'apprentissage et le test du modèle

Les fichiers build.py et itosfm.py sont communs aux deux dossiers suivants. Le premier permet de créer les échantillons d'apprentissage, validation et test ainsi que de créer le modèle. Le second sont les codes sources du modèle SFM.

--> Dossier train:
 - train.py permet d'exécuter l'apprentissage du modèle SFM, rajouter --hidden_dim=.. --freq_dim=.. pour changer les paramètres lors de la compilation

--> Dossier test:
 - test.py permet d'exécuter le test du modèle SFM sur un modèle avec les paramètres hidden_dim=10 et freq_dim=20

### Dossier LSTM
 - LSTM.ipynb permet d'exécuter le modèle LSTM appliqué au sinus

### exemple de sortie après l'exécution de test.py du dossier SFM
 ![Figure_1](https://user-images.githubusercontent.com/79654847/148119083-15200521-d9e0-4b34-8849-22059d1c1505.png)
 
### exemple de sortie après l'exécution de LSTM.ipynb du dossier LSTM
 ![sinus1](https://user-images.githubusercontent.com/79654847/148120923-77aafcd7-35ac-4c02-afc4-b6f1b49d7097.png)
 
 ## Vérification des résultats du projet
 --> Dossier Modèles:
  - 3 modèles utilisés pour chaque fichier .py

--> fichier lstm_papier.py: reproduit le modèle LSTM du papier avec des performances similaires

### sortie de l'exécution
![LSTM_papier](https://user-images.githubusercontent.com/79654847/148122097-e77849ce-415c-4003-91e7-a5afdb3b7b37.png)

--> fichier lstm_ameliore.py: utilisation d'un modèle LSTM avec de meilleurs résultats que celui du papier

### sortie de l'exécution
![LSTM_ameliorer](https://user-images.githubusercontent.com/79654847/148122139-21c181c1-6677-4a20-ab85-1a252e4726e5.png)

--> fichier sfm_papier.py: reproduit le modèle SFM du papier avec des performances similaires

### sortie de l'exécution
![SFM](https://user-images.githubusercontent.com/79654847/148122201-8d2cedab-935e-49b7-a154-e33ec38a2637.png)





