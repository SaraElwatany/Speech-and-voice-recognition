from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from flask import Flask , send_file , render_template
import io 
import base64 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import seaborn as sns

fig,ax = plt.subplots(figsize=(6,6))
# ax=sns.set_style(style='darkgrid')
data = pd.read_csv('Feature_Extraction.csv')

inputs=data.drop(columns ="Name")
target=data['Name']

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3,random_state=0) # 70% training and 30% test

sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

clf =SVC(kernel='linear', random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
# print(cm)
X_set,y_set = X_train, y_train
X1,X2=np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
np.arange(start = X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

# plt.contourf(X1,X2,clf.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
# alpha=0.75, cmap= ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i ,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
    c=ListedColormap(('red','green'))(i) , label=j)

plt.title('svm(train)')
plt.xlabel('inputs')
plt.ylabel('target')
plt.legend()
plt.show()

X_set,y_set = X_test, y_test
X1,X2=np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
np.arange(start = X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i ,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
    c=ListedColormap(('red','green'))(i) , label=j)

plt.title('svm(test)')
plt.xlabel('inputs')
plt.ylabel('target')
plt.legend()
plt.show()

