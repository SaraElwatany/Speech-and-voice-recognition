from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
# from flask import Flask , send_file , render_template
import io 
import base64 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import seaborn as sns

# fig,ax = plt.subplots(figsize=(6,6))
# ax=sns.set_style(style='darkgrid')
data = pd.read_csv('speech_features.csv')

inputs=data.drop(columns ="result")
target=data['result']

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3,random_state=0) # 70% training and 30% test

sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

clf =SVC(kernel='linear', random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
pickle.dump(clf, open("m2.pkl","wb"))           # saving model

# print(cm)