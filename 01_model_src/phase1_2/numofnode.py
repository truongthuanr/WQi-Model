#------------------------------------------------------------------------------------------#
# Purpose: loop for number of ramdom train/test to calculate average error.
#
#-----------------------------------------------------------------------------------------#


from helper import consolelog
# from config import columns, input_col, categorical_col, output_column, categorical_usecol
# from config import output_folder
# from config import zscore_lim


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error,\
                            mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import clone_model
from sklearn.linear_model import SGDRegressor
from keras.models import Sequential
from sklearn import svm
import matplotlib


columns = ['Date', 
           'Season', 
           'Vụ nuôi', 'module_name', 'ao', 
           'Ngày thả', 'Time','Nhiệt độ', 'pH', 'Độ mặn', 
           'TDS', 'Độ đục', 'DO', 'Độ màu', 'Độ trong','Độ kiềm', 
           'Độ cứng',
           'Loại ao',
           'Công nghệ nuôi', 
           'area', 
           'Giống tôm',
           'Tuổi tôm', 
           'Mực nước', 'Amoni', 
           'Nitrat', 'Nitrit', 'Silica',
            #  'Canxi', 'Kali', 'Magie'
             ]


input_col = [
     # thông số đặc tính
    'Season', 
    'Loại ao', 
    'Công nghệ nuôi', 
    'Giống tôm',  
    'Mực nước',
    'Tuổi tôm',
     'area', 
    'Nhiệt độ', 'pH', 'DO', # thông số bắt buộc
    # 'Ngày thả',
    'Độ mặn', 
    'TDS', 
    'Độ đục',
    'Độ trong',
    #phase 2 
    'Độ cứng',
    'Độ màu',
    # 'Silica',
    ]

output_folder = "output"



categorical_col = ['Date',
                   'Season', 
                   'Loại ao', 
                   'Công nghệ nuôi', 
                   'Giống tôm',
                   'units']

categorical_usecol = [
    'Season', 
    'Loại ao', 
    'Công nghệ nuôi', 
    'Giống tôm'
    ]

# output_column = ['TAN', 'Nitrat', 'Nitrit', 'Silica', 'Canxi', 'Kali', 'Magie', 'Độ kiềm', 'Độ cứng']
output_column = ['Độ kiềm']
zscore_lim =  3


def plot_result3x3(y_test,y_pred,output_column):
    fig = plt.figure(figsize = [8,8])
    for i,col in enumerate(output_column):
        plt.subplot(3,3,i+1)
        plt.scatter(x=y_test[:,i],
                    y=y_pred[:,i],
                    marker = 'X',
                    lw=0.5,
                    s=10,
                    color=matplotlib.cm.tab20.colors[i])
        lim = [plt.xlim()[0],plt.xlim()[1]]
        plt.plot(lim,lim,
                color='grey')
        plt.title(col+f" MAE: {mean_absolute_error(y_test[:,i],y_pred[:,i]):.3f}",
                fontsize='small')
        
    fig.suptitle("RandomForest")       
    plt.tight_layout()
    # plt.show()
    suffix = datetime.strftime(datetime.now(),"%y%m%d-%H%M%S")
    plt.savefig(os.path.join(output_folder,f"randomforest_{suffix}.png"))
    

def plot_result(y_test,y_pred,output_column,modelname: str):
    fig = plt.figure(figsize = [8,8])
    for i,col in enumerate(output_column):
        # plt.subplot(3,3,i+1)
        plt.scatter(x=y_test[:,i],
                    y=y_pred[:,i],
                    marker = 'X',
                    lw=0.5,
                    s=10,
                    color=matplotlib.cm.tab20.colors[0])
        lim = [plt.xlim()[0],plt.xlim()[1]]
        plt.plot(lim,lim,
                color='grey')
        plt.title(col,
                    fontsize='small')
        error_text = f"RMSE: {root_mean_squared_error(y_test[:,i],y_pred[:,i]):.3f}" + "\n" +\
                     f"MAE: {mean_absolute_error(y_test[:,i],y_pred[:,i]):.3f}"+ "\n" +\
                     f"MAPE: {mean_absolute_percentage_error(y_test[:,i],y_pred[:,i])*100:.3f}%"+ "\n" +\
                     f"R2 Score: {r2_score(y_test[:,i],y_pred[:,i]):.3f}"
                     
        plt.text(x=lim[0]+(lim[1]-lim[0])*0.1,
                 y=lim[0]+(lim[1]-lim[0])*0.9,
                 s=error_text)
        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        
    fig.suptitle(modelname)       
    plt.tight_layout()
    # plt.show()
    suffix = datetime.strftime(datetime.now(),"%y%m%d-%H%M%S")
    plt.savefig(os.path.join(output_folder,f"{modelname}_{suffix}.png"))


    # print(y_test[:,0])
    # print(y_pred[:,0])

    fig2 = plt.figure(figsize = [8,8])
    ind = np.argsort(y_test[:,0])
    # print(ind)
    plt.plot(y_test[ind],label="True value")
    plt.plot(y_pred[ind],label="Predict value",
             ls="",
             marker="x")
    plt.legend()
    plt.xlabel("Samples")
    plt.title(f"Ground Truth and Predicted value - {modelname}")
    plt.savefig(os.path.join(output_folder,f"{modelname}_GroundTruth_{suffix}.png"))

def plot_history(his):
    fig = plt.figure(figsize = [8,8])
    plt.plot(his['loss'], label="loss")
    plt.plot(his['r2_score'], label="r2_score")
    plt.plot(his['val_loss'], label="val_loss")
    plt.plot(his['val_r2_score'], label="val_r2_score")
    plt.legend()

    plt.title(f"HISTORY")
    suffix = datetime.strftime(datetime.now(),"%y%m%d-%H%M%S")
    plt.savefig(os.path.join(output_folder,f"ANN_history_{suffix}.png"))




def readdata() -> pd.DataFrame:
    consolelog("Read data!")
    df = pd.read_csv("./../../../dataset/data_4perday_cleaned.csv", usecols=columns)

    # Rename columns Amoni -> Tan 
    df.rename({'Amoni':'TAN'},axis=1,inplace=True)

    return df


def datacleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Cột 'Mực nước', thay các ô có giá trị = 0 thành NaN
    # Thay các giá trị NaN bằng median của cột
    df.loc[df['Mực nước']==0,'Mực nước']=np.NaN
    df['Mực nước'].fillna(df['Mực nước'].median(), inplace=True)

    # Drop cols "Time"
    df.drop(['Time'], axis=1,inplace=True)

    # Format "Date"
    df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y')

    # convert 'Tuoi tom' column to numeric
    # cell which is not able to convert to float (#REF) will be fill as NaN
    df['Tuổi tôm'] = df['Tuổi tôm'].apply(lambda x: 
                                        int(float(x)) if str(x).replace('.','',1).isnumeric() 
                                        else np.NaN)
    
    df['units'] = df.apply(lambda x:  f"{x['Vụ nuôi'].replace(' ','')}-{x['module_name']}-{x['ao']}" ,axis=1)
    df.drop(['Vụ nuôi','module_name','ao'],axis=1,inplace=True)

    # Sort data by unit and date
    df.sort_values(['units','Date'],inplace=True)

    # Drop NA
    df.dropna(axis=0,inplace=True)

    return df



def preprocessingdata(df: pd.DataFrame)-> pd.DataFrame:
    df_num = df.drop(categorical_col,axis=1)

    df1 = df[(np.abs(stats.zscore(df_num))<zscore_lim).all(axis=1)].copy()
    fig= plt.figure(figsize=(10,5))
    ax = sns.boxplot(df1)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    # plt.show()
    plt.savefig(os.path.join(output_folder,"boxplot1.png"))


    df1 = df1[input_col+output_column].copy()
    df1.reset_index(drop=True,inplace=True)

    df1.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

    consolelog("Plot data!")
    plt.figure(figsize=(10,5))
    sns.boxenplot(df1)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    # plt.show()
    plt.savefig(os.path.join(output_folder,"boxplot2.png"))


    consolelog("One hot encorder")
    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(df1[categorical_usecol])
    oh_df = pd.DataFrame(oh_enc.transform(df1[categorical_usecol]),
                     columns=oh_enc.get_feature_names_out()
                    )
    print(oh_df.columns)
    df1 = pd.concat([oh_df,df1],axis=1)
    df1.drop(categorical_usecol,inplace=True,axis=1)

    return df1


def ANNModel(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.33, random_state=42)

    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train_tf = X_sc.transform(X_train)
    print(f"{X_train_tf.shape}")
    y_sc = StandardScaler()
    y_sc.fit(y_train)
    y_train_tf = y_sc.transform(y_train)
    print(f"{y_train_tf.shape}")
    y_test=y_test.to_numpy()

    with open("./output/ANN_numofnode.csv","w+") as f:
        f.write("Set, MRSE, MAE, MAPE(%), R2Score\n")

        for node in range(1,21):

            print(f"----- Loop {node} ------")   

            # define the model
            model1 = Sequential()
            model1.add(Input(shape=(21,)))
            model1.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(0.1))
            model1.add(Dense(16+node,kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(0.1))
            model1.add(Dense(8, kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(0.1))
            model1.add(Dense(1))
            model1.compile(loss='mae', 
                        optimizer='nadam',
                        metrics=['accuracy','r2_score']
                        ) 

            model1.compile(loss='mae', 
                optimizer='nadam',
                metrics=['accuracy','r2_score']
                )
            history = model1.fit(X_train_tf,y_train_tf,
                        epochs=200,
                        batch_size=32,
                        verbose=0,
                        validation_split=0.2)
            # plot_history(history.history)

            y_pred = model1.predict(X_sc.transform(X_test))
            y_pred = np.reshape(y_pred,(-1,1))
            
            
            y_pred=y_sc.inverse_transform(y_pred)
            i=0
            RMSE = f"{root_mean_squared_error(y_test[:,i],y_pred[:,i]):.3f}"
            MAE  = f"{mean_absolute_error(y_test[:,i],y_pred[:,i]):.3f}"
            MAPE = f"{mean_absolute_percentage_error(y_test[:,i],y_pred[:,i])*100:.3f}"
            R2Score =  f"{r2_score(y_test[:,i],y_pred[:,i]):.3f}"
            f.write(f"{node+57},{RMSE},{MAE},{MAPE},{R2Score}\n")

            # plot_result(y_test=y_test.to_numpy(),y_pred=y_sc.inverse_transform(y_pred),output_column=output_column,modelname="ANN")


def noname():
    df = readdata()
    # print(df.head())

    df = datacleaning(df)
    # print(df.head())

    df = preprocessingdata(df)
    # print(df.columns)
    X = df.drop(output_column, axis=1)
    print()
    print(f"{X.columns=}")
    y = df[output_column]
    # get_random_grid()


    ANNModel(X,y)


if __name__ == "__main__":
    consolelog("Running main Program")
    noname()