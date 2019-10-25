import pandas as pd
import numpy as np
import keras as ks
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.regularizers import l2, l1_l2
from keras.callbacks import History
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy import stats
from scipy.stats import chisquare
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
"""
Autore: Lorenzo Famiglini
Matricola: 838675
"""

"""
PREPARAZIONE DATI
"""
df_train = pd.read_csv("/Users/lorenzofamiglini/Desktop/MsC_2_anno/Advanced_ML/Assignment1_AML/train.csv")
df_train.describe()

df_train[['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].hist(figsize = (20,20))

#g = sns.pairplot(df_train.iloc[:,0:5])
"Removing outliers per valori al di fuori del 5 e 95 percentile"
#P = np.percentile(df_train.LIMIT_BAL, [5, 95])
#df_train = df_train[(df_train.LIMIT_BAL > P[0]) & (df_train.LIMIT_BAL < P[1])]
df_test = pd.read_csv("/Users/lorenzofamiglini/Desktop/MsC_2_anno/Advanced_ML/Assignment1_AML/test.csv")
df_test.describe()
Y = df_train[["default.payment.next.month"]]
X = df_train.iloc[:,0:23]

#Plot target variable:
"""
Class imbalanced problem
Default payment: 1=yes, 0=no
"""
sns.countplot(x="default.payment.next.month", data=Y)
#CHECK MISSING VALUES
df_train.info() #no missing VALUES
df_train
#Check correlation between default payment and age:
np.corrcoef(Y["default.payment.next.month"],X.AGE)
np.corrcoef(df_train.iloc[:,4:22])
"""
Discretizziamo l'età in tre principali fasce in modo tale da vedere se la variabile discretizzata fornisce un informazione in più rispetto a quella originale
Una volta discretizzato applichiamo il test del Chi-squared in modo tale da vedere se sussiste una dipendenza o meno.
"""

#Discretization age in three main levels:

X["Discr_age"] = pd.qcut(X["AGE"], 3, labels=["young", "adult", "old"])
sns.countplot(x="Discr_age", data=X)

#Chi-squared test
contingency_table = pd.crosstab(
    X["Discr_age"],
    Y["default.payment.next.month"],
    margins = True)
contingency_table
stats.chi2_contingency(contingency_table)
chisquare(contingency_table)
#Ipotesi nulla rifiutata --> sussiste dipendenza. Questo significa che la variabile discretizzata potrebbe spiegare meglio della variabile dell'età originaria.
import category_encoders as cs
#encoder = cs.BinaryEncoder(cols=['EDUCATION'])
#X = encoder.fit_transform(X)
X = pd.get_dummies(X, columns=["EDUCATION", "Discr_age","SEX","MARRIAGE"])
X = X.drop(columns = ["AGE"])
"""
Train test split
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,stratify=Y,test_size=0.20)

"""
min max normalizziamo il test rispetto al train
"""
def minMax(x):
    return pd.DataFrame(index=['min','max'],data=[x.min(),x.max()])
min_max = minMax(X_train.iloc[:,0:19])

for i in min_max:
    min = min_max.loc['min',i]
    max = min_max.loc['max', i]
    X_test[i] = X_test[i].apply(lambda x: (x - min)/(max-min))

"""
Normalization e over_sampling
"""
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
mms = MinMaxScaler()
X_train.iloc[:,0:19] = mms.fit_transform(X_train.iloc[:,0:19])
X_train, y_train = SVMSMOTE().fit_resample(X_train, np.ravel(y_train))
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
#Shuffle data:
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train,y_train, random_state=1)
sns.countplot(x=0, data=pd.DataFrame(y_train))
X_train.shape
"""
Modello 1
"""
#class_weights = {1: 0.55,
                #0: 0.45}
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
#model.load_weights('my_model_weights.h5')
#sgd = optimizers.SGD(lr=0.001,nesterov=False)
#optimizers.Adam(learning_rate=0.1,amsgrad=False)
#Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
#LR_DECAY = 1e-3
history = History()
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu")) [35,512,128,64,128,64,128,2]
model.add(Dense(64, activation = "relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))
es = EarlyStopping(monitor='val_loss', patience = 15, min_delta=0.1)
model.compile(loss='categorical_crossentropy',optimizer=ks.optimizers.Nadam(learning_rate=0.01, beta_1=0.8, beta_2=0.99), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=256, callbacks=[history], validation_data=(X_test,y_test), verbose=1)
y_pred = model.predict(X_test,verbose=1)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred.round()))
model2.evaluate(X_test, y_test, verbose=1)[0]
model2.evaluate(X_train, y_train)[0]

"""
Salviamo i pesi
"""
model.save_weights('my_model_weights.h5')


"""
Plot accuratezza e loss
"""
#Plot accuracy
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""
ROC e AUC curve
"""
y_pred_mo1 = model.predict(X_test,verbose=1)
y_pred_mo2 = model2.predict(X_test, verbose = 1)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
fpr_keras, tpr_keras, thresholds_keras = roc_curve([np.argmax(z) for z in y_test],[np.argmax(z) for z in y_pred_mo1])
auc_keras = auc(fpr_keras, tpr_keras)

fpr_keras2, tpr_keras2, thresholds_keras2 = roc_curve([np.argmax(z) for z in y_test],[np.argmax(z) for z in y_pred_mo2])
auc_keras2 = auc(fpr_keras2, tpr_keras2)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Complex model (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

#plt.figure(1)
#plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras2, tpr_keras2, label='Simple model (area = {:.3f})'.format(auc_keras2))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


"""
Model 2
"""

"""
GridSearch numero di neuroni, layers, learning rate ec..
"""

hidd1 = [16,32,64,128]
hidd2 = [16,32,64,128]
hidd3 = [16,32,64,128]
lr = [0.1,0.01,0.001,0.0001]
n_layers = [1,2,3,4]
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
fBestModel = 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'

best_f1 = 0
new_acc = 0
for i in hidd1:
    for j in hidd2:
        for x in hidd3:
                history2 = History()
                model2 = Sequential()
                model2.add(Dense(i, input_dim=X_train.shape[1], activation="relu"))
                model2.add(Dense(j, activation="relu"))
                model2.add(Dense(x, activation="relu"))
                model2.add(Dense(2, activation="softmax"))
                model_best_f1 = 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
                es = EarlyStopping(monitor='val_loss', patience=30, verbose=0)
                best_model = ModelCheckpoint(model_best_f1, verbose=0, save_best_only=True)
                model2.compile(loss='categorical_crossentropy',optimizer = ks.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.99), metrics=['accuracy'])
                model2.fit(X_train, y_train, epochs=50, batch_size=256, callbacks=[best_model, history2, es], validation_data=(X_test,y_test), verbose=0)
                y_pred = model.predict(X_test).argmax(axis=-1)
                f1 = f1_score(y_test.argmax(axis=1),y_pred)
                acc = model2.evaluate(X_test, y_test, verbose=1)[1]
                if (f1>=best_f1 and acc > new_acc):
                    best_f1 = f1
                    best_mod = [i,j,x]
                    new_acc = acc
                print("Numero di neuroni usati in ordine di strato: " + "["+ str(i),str(j),str(x)+"]")
                print("Accuratezza sul test: "+str(new_acc),"F1 measure sul test: " +str(f1))
""""
[35 10 44 64]
f1_s = {"f1-score": [0.50, 0.49,0.51,0.5,0.5,0.52,0.51,0.53,0.50,0.51,0.5,0.5,0.5,0.52,0.51,0.48,0.49,0.51,0.51,0.50,0.49,0.51,0.51,0.50,0.51,0.53,0.50]}
f1_s = pd.DataFrame(f1_s)
sup_ci = mean + 1.96*(np.sqrt((mean*(1-mean))/len(f1_s["f1-score"])))
mean = f1_s["f1-score"].mean()
sup_ci = mean + 1.96*(np.sqrt((mean*(1-mean))/len(f1_s["f1-score"])))
low_ci = mean - 1.96*(np.sqrt((mean*(1-mean))/len(f1_s["f1-score"])))
sup_ci
low_ci
"""
best_mod
new_acc
[35,16,64,128]
"""
Analisi secondo modello
"""
from sklearn.metrics import confusion_matrix
model.evaluate(X_train, y_train)
model.evaluate(X_test, y_test)
y_pred_mod2 = model2.predict(X_test)
print(classification_report(np.argmax(y_test, axis = 1), np.argmax(y_pred_mod2.round(), axis = 1)))

y_pred_mod2 = model2.predict_classes(X_train)
y_pred_mod1 = model.predict_classes(X_train)

cm = confusion_matrix(np.argmax(y_test, axis = 1),y_pred_mod1)

#Plot accuracy
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#Plot loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




"""
Testiamo i risultati sui i veri dati di test
"""

X_val = df_test.iloc[:,0:23]
X_val["Discr_age"] = pd.qcut(X_val["AGE"], 3, labels=["young", "adult", "old"])
X_val = pd.get_dummies(X_val, columns=["EDUCATION", "Discr_age","SEX","MARRIAGE"])
for i in min_max:
    min = min_max.loc['min',i]
    max = min_max.loc['max', i]
    X_val[i] = X_val[i].apply(lambda x: (x - min)/(max-min))
X_val = X_val.drop(columns = ["AGE"])
y_pred_mod1 = model.predict_classes(X_val)

y_pred_mod2 = model2.predict_classes(X_val)

risultato_finale = pd.DataFrame(np.bincount(y_pred_mod1),columns=["0"])
risultato_finale = risultato_finale.reset_index()

sns.set(style="darkgrid")
sns.countplot(x="index",y =  data=risultato_finale)
prediction_Final = model.predict_classes(X_val)
np.savetxt('Lorenzo_Famiglini_838675_score1.txt', prediction_Final)
with open("Lorenzo_Famiglini_838675_score1.txt", "wb") as f:
    f.write(prediction_Final)
    np.savetxt(f, prediction_Final.astype(int), fmt='%i', delimiter=",")
    f.close()
