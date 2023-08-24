import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.utils import shuffle
from keras import backend as K
import random
import matplotlib.pyplot as plt
import os
from keras.backend import manual_variable_initialization
from sklearn.metrics import mean_squared_error

data_len=10
rnn_size=data_len
predict_len=1
val_percent=0.8


#df_blockchain = pd.read_csv(os.path.join("data", "df_blockchain.csv"), delimiter=";")
df = pd.read_csv("data/df_blockchain.csv",sep=";",names=["date","price","transaction_per_block","confirmation_time","hash_rate","difficulty","miner_revenue","volume","blocks_size","avg_block_size","transaction_fees","transaction_fees_drop","cost_per_transaction_percent","cost_per_transaction_drop","n_unique_addresses","n_transactions","n_transactions_drop","n_transactions_drop2","out_volume","volume_drop","volume_drop2","total_bitcoins","market_cap"])
#df = pd.read_csv("data/df_blockchain.csv",sep=";",names=["date","price"])

df.drop([0],inplace=True)

for column in df.columns:
    if "drop" in column:
        df.drop(columns=[column],inplace=True)
    
df.dropna(axis="columns", how='any',inplace=True)

df.drop(['date'],axis=1,inplace=True)

# get columns (You can add more columns to analyse results)

columns = df.columns
print(columns)

#--------------------- Scaling des données par colonne (MinMax)

dataset=df
dataset = dataset.values.reshape(-1,len(columns))

scaler = MinMaxScaler()
scaler_price=MinMaxScaler()
dataset=np.transpose(dataset)

price_scaled=scaler_price.fit_transform(dataset[0].reshape(-1,1)).reshape(1,-1)[0]
dataset[0]=price_scaled

dataset=np.transpose(dataset)
dataset=scaler.fit_transform(dataset)

scaler.fit_transform(dataset)





def process_data(data, rnn_size=rnn_size, target_id=0, columns_size=len(columns)):
    X = []
    y = []
    for i in range(len(data)-rnn_size):
        X.append(data[i:i+rnn_size,:])
        y.append([data[i+rnn_size,0]])
    
    return np.array(X).astype(np.float32).reshape((-1,rnn_size,columns_size)), np.array(y).astype(np.float32)


X,Y=process_data(dataset)

X_tot=np.copy(X)
Y_tot=np.array([a for a in Y])


#Définir les données de test (à prédire)
Y_test=Y[-30:]
X_test=X[-30:]  
print(X_test)

#Définir les données d'entrainement

X_train=X[:-30]
Y_train=Y[:-30]

#Mélange pour un entrainement uniforme
X_train,Y_train=shuffle(X_train,Y_train)

list_rmse_grid=[]

def create_model_base():
    model = Sequential()

    model.add(LSTM(128, input_shape=(data_len,len(columns)), activation = 'sigmoid', return_sequences = True))
    model.add(LSTM(128,activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='sigmoid'))
    model.add(Dropout(0.2)) #Pour empêcher un overfitting
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    return model


#Charge le meilleur model trouvé par grid_search
model=tf.keras.models.load_model("best\model188")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


#Calcule la RMSE sur les données test et permet de sauvegarder les graphes
def RMSE_model(model,afficher=True,epoch=None,coords="0_0_0_0"):
    y_pred=np.array([[a[0]] for a in X_test[0]])
    y_real=np.array([[a[0]] for a in X_test[0]])
    pred=model.predict(X_test)
    real=Y_test
    pred=np.concatenate((y_pred,pred),axis=0)
    real=np.concatenate((y_real,real),axis=0)
    pred=scaler_price.inverse_transform(pred)
    real=scaler_price.inverse_transform(real)
    c=rmse(pred,real)

    pred_plot=[a[0] for a in pred]
    real_plot=[a[0] for a in real]
    plt.plot(pred_plot,label="Prédiction")
    plt.plot(real_plot,label="Vraies valeurs")
    plt.axvline(x=data_len-1,color='r')
    plt.title("RMSE = " + str(c))
    plt.legend()
    if afficher:
        plt.show()
    else:
        plt.savefig(f"data/prediction_epoch_{coords}_{epoch}.png")
        plt.clf()
    return c

RMSE=220
list_rmse=[]

#Adaptation automatique du learning rate et sauvegarde des prédictions.
class custom_callback(tf.keras.callbacks.Callback): #A chaque fin de epoch, lance la fonction de test
    def on_epoch_end(self,epoch,logs=None):
        self.model.optimizer.learning_rate=self.model.optimizer.learning_rate*0.99
        e_model=self.model
        print("learning rate now set to "+str(self.model.optimizer.learning_rate))
        RMSE_e_model = RMSE_model(e_model,afficher=False,epoch=epoch)
        list_rmse.append(RMSE_e_model)
        print(f"Epoch {epoch}, RMSE = {RMSE_e_model}")
        if RMSE_e_model<RMSE:
            e_model.save(f"models\model{epoch}")

#Callback nécessaire pour le grid_search, permet une mémoire du modèle
class grid_custom_callback(tf.keras.callbacks.Callback):
    def __init__(self,list_rmse_grid,coords,decay=0.95):
        super(grid_custom_callback,self).__init__()
        self.list_rmse_grid=list_rmse_grid
        self.coords=coords
        self.decay=decay
    def on_epoch_end(self,epoch,logs=None):
        e_model=self.model
        RMSE_e_model = RMSE_model(e_model,afficher=False,epoch=epoch,coords=self.coords)
        self.model.optimizer.learning_rate=self.model.optimizer.learning_rate*self.decay
        #if RMSE_e_model<min(self.list_rmse_grid):
        #    e_model.save("data\model_"+self.coords)

        self.list_rmse_grid.append(RMSE_e_model)


#A ne pas lancer si on a chargé un modèle déjà entrainé.
def train_model(model):
    model.summary()
    opt=tf.keras.optimizers.Adam()
    model.compile(optimizer = opt, loss ="mse",)
    manual_variable_initialization(False) #Empêche de sauvegarder un modèle aléatoire
    history=model.fit(X_train,Y_train, epochs=200,callbacks=[custom_callback()])
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()


print(RMSE_model(model))

#Grid search sur les paramètres du modèles, /!\ très long
def grid_search(lr_min,lr_max,nb_lr,max_epoch,optimizers=[tf.keras.optimizers.Adam()],lstm=[[128]],dense=[[128]]):
    list_lr=np.geomspace(lr_min,lr_max,nb_lr)
    dic_rmses={}
    for i in range(len(lstm)):
        for j in range(len(dense)):
            for k in range(len(list_lr)):
                for l in range(len(optimizers)):
                    rmse_callback=grid_custom_callback([10000],f"{i}_{j}_{k}_{l}",decay=1)
                    model_grid=Sequential()
                    for lstm_index in range(len(lstm[i])):
                        if lstm_index==0 and len(lstm[i])>1:
                            model_grid.add(LSTM(lstm[i][lstm_index], input_shape=(data_len,len(columns)), activation = 'sigmoid', return_sequences = True))
                        elif lstm_index==0:
                            model_grid.add(LSTM(lstm[i][lstm_index], input_shape=(data_len,len(columns)), activation = 'sigmoid', return_sequences = False))
                        else:
                            model_grid.add(LSTM(lstm[i][lstm_index],activation="sigmoid"))
                    for dense_index in range(len(dense[j])):
                        model_grid.add(Dense(dense[j][dense_index]))
                    model_grid.add(Dense(1))
                    opt = optimizers[l]
                    opt.learning_rate = list_lr[k]
                    model_grid.compile(optimizer=opt,loss='mse')
                    es=EarlyStopping('val_loss',patience=2*max_epoch//3,mode="min",verbose=1,min_delta=10) #permet de s'arrêter quand la validation remonte de 10%, utile surtout pour les faibles learning rates
                    #model_grid.fit(X_train,Y_train,epochs=max_epoch,validation_data=(X_test,Y_test),callbacks=[rmse_callback,es])
                    model_grid.fit(X_train,Y_train,epochs=max_epoch,validation_split=0.2,callbacks=[rmse_callback,es])
                    dic_rmses[(i,j,k,l)]=min(rmse_callback.list_rmse_grid)

    return dic_rmses    


#print(grid_search(1e-5,1e-3,10,120,optimizers=[tf.keras.optimizers.RMSprop(),tf.keras.optimizers.Adam()]))
