import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

data_file = pd.read_csv("Iris.csv")#read data
data_file = data_file.drop(['Id'], axis=1)#gets rid of Id column
target = data_file['Species']#only species would be shown

#s = set()#object that will store smth but it doesnt take into account repetitions
#for val in target:
#    s.add(val)#checks what is in target then saves it into dictionary
#s = list(s)#creates a list
##below it choses only values that are from 0-99 rest is being deleted
#rows = list(range(100,150))
#data_file = data_file.drop(data_file.index[rows])

#print(data_file)

#x = data_file['SepalLengthCm']
#y = data_file['PetalLengthCm']
#setosa_x = x[:50]
#setosa_y = y[:50]
#versicolor_x = x[50:]
#versicolor_y = y[50:]



#plt.figure(figsize=(8,6))
#plt.scatter(setosa_x,setosa_y,marker='+',color='green')
#plt.scatter(versicolor_x,versicolor_y,marker='_',color='red')
#plt.show()

#_______________________________

data_file = data_file.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
Y = []#Y has either Iris-setosa or anything else
for Val in target:
    if(Val == 'Iris-setosa'):
        Y.append(-1)
    else:
        Y.append(1)

data_file = data_file.drop(['Species'], axis=1)
X = data_file.values.tolist()#two remaining columns are transformed into list of lists where each inner list-like-cell is a single row

X,Y = shuffle(X,Y)#creates random order of X and Y related to its X

x_train = []
y_train = []
x_test = []
y_test = []
#below I split the data_file into:90% is training & 10% is testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train) #parameters of length&width
y_train = np.array(y_train)#y_train relates to exact x_train with information whether it is Iris-setosa or not
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(135,1)
y_test = y_test.reshape(15,1)

#SVM

train_f1 = x_train[:,0]#[ first_row:last_row , column_0 ]
train_f2 = x_train[:,1]#-^

train_f1 = train_f1.reshape(135,1)
train_f2 = train_f2.reshape(135,1)

#weights initialization
w1 = np.zeros((135,1))
w2 = np.zeros((135,1))

epochs = 1
alpha = 0.0001

while(epochs < 10000):
    y = w1*train_f1 + w2*train_f2
    prod = y*y_train #classification measurement
    count = 0
    for val in prod:
        if(val >= 1):#properly classified
            cost = 0
            w1 = w1 - alpha*(2 * 1/epochs * w1)#what is in the bracket is LAMBDA coefficient
            w2 = w2 - alpha*(2 * 1/epochs * w2)
        else:#badly classified
            cost = 1 - val

            w1 = w1 + alpha*(train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
            w2 = w2 + alpha*(train_f2[count]*y_train - 2 * 1/epochs * w2)
        count += 1
    epochs += 1

index = list(range(15,135))
w1 = np.delete(w1,index)
w2 = np.delete(w2,index)

w1 = w1.reshape(15,1)
w2 = w2.reshape(15,1)

#prepraing test
test_f1 = x_test[:,0]
test_f2 = x_test[:,1]

test_f1 = test_f1.reshape(15,1)
test_f2 = test_f2.reshape(15,1)

#predict
y_pred = w1*test_f1 + w2*test_f2
predictions = []

for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

print(accuracy_score(y_test,predictions))
#comparison with built-in function
clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
y_pred1 = clf.predict(x_test)
print(accuracy_score(y_test,y_pred1))