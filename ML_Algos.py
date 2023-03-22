#import libraries 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report

#import dataset: 
creditcarddb = pd.read_csv(r"C:\Users\lazra\OneDrive\Bureau\TP ML\creditcard.csv")


#logistic regression algorithm :

#On extrait les valeurs de toutes les colonnes de la base de données , a l'exception de la derniere colonne , et on le stocke dans une variable X :
X = creditcarddb.iloc[:,:-1].values
y = creditcarddb.iloc[:,-1].values

#On divise les données selon deux catégories, données d'entrainement et données de test 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

#On met en place notre modèle ( qui est dans ce cas l'algorithme de la regression logistique )
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred2 = model.predict(X_train)

#On mesure le score de performance:
from sklearn.metrics import accuracy_score
score_train = accuracy_score(y_train, y_pred2)
print("\n Score de performance d'entrainement: ")
print(score_train)

score_test = accuracy_score(y_test, y_pred)
print('\n Score de performance test: ')
print(score_test)

#On met en place la matrice de confusion:
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("matrice de confusion : ",cm)


#SVM algorithm:

model = SVC()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
y_pred2=model.predict(X_train)

#On mesure les performances:
score_train=accuracy_score(y_train,y_pred2)
#print("score de performance d'entrainement:",score_train)
score_test=accuracy_score(y_test,y_pred)
print("score de performance de test:",score_test)

#et on met en place la matrice de confusion:
cm=confusion_matrix(y_test,y_pred)
print("matrice de confusion : ",cm)


#DNN algorithm : 
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import layers

#import database:
creditcarddb = pd.read_csv(r"C:\Users\lazra\OneDrive\Bureau\TP ML\creditcard.csv")

X = creditcarddb.iloc[:,:-1]
y = creditcarddb.iloc[:,-1]

# On divise les données selon deux catégories, données d'entrainement et données de test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Definir notre modele :
feature_columns = []
for feature_name in X.columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name))

dnnmodel = tf.keras.models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compiler notre modèle
dnnmodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = dnnmodel.fit(X_train, y_train,
                    batch_size=50,
                    epochs=25,
                    validation_data=(X_test, y_test))

# Evaluation du modèle:
score = dnnmodel.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

