from helper import consolelog
from config import columns, input_col, categorical_col, categorical_usecol
from config import output_folder
from config import zscore_lim



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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.linear_model import SGDRegressor
from keras.models import Sequential
from sklearn import svm
import matplotlib


output_column = ['Độ kiềm ngày tiếp theo']

todayparam = ['Độ kiềm']





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
    ind = np.argsort(y_pred[:,0])
    # print(ind)
    plt.plot(y_pred[ind],label="True value")
    plt.plot(y_test[ind],label="Predict value",
             ls="",
             marker="x")
    plt.legend()
    plt.xlabel("Samples")
    plt.title(f"Ground Truth and Predicted value - {modelname}")
    plt.savefig(os.path.join(output_folder,f"{modelname}_GroundTruth_{suffix}.png"))


def readdata() -> pd.DataFrame:
    consolelog("Read data!")
    df = pd.read_csv("./../../dataset/data1.csv", usecols=columns)

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
                                        int(float(x)) if x.replace('.','',1).isnumeric() 
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

    # # lastday columns
    # # Tạo thêm cột lastday-column chứa dữ liệu của ngày hôm trước
    # # trong mỗi hàng, khởi tạo các cột với giá trị NaN
    # # then, drop the NaN row
    # ld_column = [f"ld_{col}" for col in output_column]
    # df[ld_column] = np.NaN

    # # Copy data của ngày hôm trước cho mỗi row
    # unit_l = list(df['units'].unique())
    # for unit in unit_l:
    #     df.loc[df['units']==unit,ld_column] = df.loc[df['units']==unit,output_column].shift(1).to_numpy(copy=True)
    # df.dropna(axis=0,inplace=True)



    # nextday columns
    # Tạo thêm cột nextday-column chứa dữ liệu của ngày tiếp theo (output)
    
    df[output_column] = np.NaN

    # Copy data của ngày hôm trước cho mỗi row
    unit_l = list(df['units'].unique())
    for unit in unit_l:
        df.loc[df['units']==unit,output_column] = df.loc[df['units']==unit,todayparam].shift(-1).to_numpy(copy=True)
    df.dropna(axis=0,inplace=True)
    print(df.head())


    #-----------------------------------------------------#
    #
    #-----------------------------------------------------#

    df = df[input_col + todayparam + output_column].copy()
    df.reset_index(drop=True,inplace=True)

    df1.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

    consolelog("Plot data!")
    plt.figure(figsize=(10,5))
    sns.boxenplot(df)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    # plt.show()
    plt.savefig(os.path.join(output_folder,"boxplot2.png"))


    consolelog("One hot encorder")
    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(df[categorical_usecol])
    oh_df = pd.DataFrame(oh_enc.transform(df[categorical_usecol]),
                     columns=oh_enc.get_feature_names_out()
                    )
    print(oh_df.columns)
    df = pd.concat([oh_df,df],axis=1)
    df.drop(categorical_usecol,inplace=True,axis=1)

    return df


def RandomForest_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.33, random_state=42)
    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train_tf = X_sc.transform(X_train)

    y_sc = StandardScaler()
    y_sc.fit(y_train)
    y_train_tf = y_sc.transform(y_train)

    consolelog("Create MultiOutput RandomForeest")
    
    rf = RandomForestRegressor(n_estimators=200,
                               max_depth=50,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               max_features='sqrt', 
                               random_state=0,
                               bootstrap=True,
                               verbose=1,)

    consolelog("Training...")
    rf.fit(X_train_tf,np.reshape(y_train_tf,(-1)))

    rf.score(X_sc.transform(X_test),y_sc.transform(y_test))
    y_pred = rf.predict(X_sc.transform(X_test))
    y_pred = np.reshape(y_pred,(-1,1))

    # for i in range(y_pred.shape[1]):
    #     plt.subplot(3,3,i+1)
    #     plt.scatter(y_sc.transform(y_test)[:,i],y_pred[:,i])

    # print(f"{y_sc.inverse_transform(np.reshape(y_pred,(1,-1))).shape=}")
    # print(f"{type(y_sc.inverse_transform(np.reshape(y_pred,(1,-1))))=}")
    print(f"{type(y_pred)=}")
    print(f"{y_pred.shape=}")

    
    print(f"{y_test.to_numpy().shape=}")
    print(f"{type(y_test.to_numpy())}")


    # plot_result(y_sc.transform(y_test),y_pred, output_column,modelname="RandomForest")
    plot_result(y_test=y_test.to_numpy(),
                y_pred=y_sc.inverse_transform(y_pred), 
                output_column=output_column,
                modelname="RandomForest")
    

    # # Random searchCV
    # rf = RandomForestRegressor()
    # random_grid = get_random_grid()
    # rf_random = RandomizedSearchCV(estimator = rf, 
    #                                param_distributions = random_grid, 
    #                                n_iter = 100, 
    #                                cv = 3, 
    #                                verbose=2, 
    #                                random_state=42, 
    #                                n_jobs = -1)
    # rf_random.fit(X_train_tf,np.reshape(y_train_tf,(-1)))
    # print(rf_random.best_params_)

def SVRModel(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.33, random_state=42)
    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train_tf = X_sc.transform(X_train)

    y_sc = StandardScaler()
    y_sc.fit(y_train)
    y_train_tf = y_sc.transform(y_train)

    consolelog("Create SGD Regressor")

    SVRreg = svm.SVR(kernel='rbf',epsilon=0.2,C=1)
    # print(f"{X_train.shape=} {y_train.shape=}")
    SVRreg.fit(X_train_tf,np.reshape(y_train_tf,(-1)))

    y_pred = SVRreg.predict(X_sc.transform(X_test))
    y_pred = np.reshape(y_pred,(-1,1))


    plot_result(y_test=y_test.to_numpy(),
            y_pred=y_sc.inverse_transform(y_pred), 
            output_column=output_column,
            modelname="SVRModel")
    
    # # Random searchCV
    # SVRreg = svm.SVR()
    # random_grid = getsvrgrid()
    # rf_random = RandomizedSearchCV(estimator = SVRreg, 
    #                                param_distributions = random_grid, 
    #                                n_iter = 100, 
    #                                cv = 3, 
    #                                verbose=2, 
    #                                random_state=42, 
    #                                n_jobs = -1)
    # rf_random.fit(X_train_tf,np.reshape(y_train_tf,(-1)))
    # print(rf_random.best_params_)


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

    # define the model
    model1 = Sequential()
    model1.add(Input(shape=(19,)))
    model1.add(Dense(15, kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(10,kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(5, kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(1))
    model1.compile(loss='mae', 
                   optimizer='nadam',
                   )
    model1.summary()

    model1.fit(X_train_tf,y_train_tf,
            epochs=200,
            batch_size=32,
            verbose=True)

    y_pred = model1.predict(X_sc.transform(X_test))
    plot_result(y_test=y_test.to_numpy(),y_pred=y_sc.inverse_transform(y_pred),output_column=output_column,modelname="ANN")


def get_random_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    # print(random_grid)
    return random_grid


def getsvrgrid():
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [0.1, 1.0, 10.0, 100.0], 
    epsilon = [0.01, 0.1, 0.2, 0.5]



    random_grid = {'kernel':kernel,
                   "C":[0.1, 1.0, 10.0, 100.0],
                   "epsilon":epsilon}
    
    return random_grid


def noname():
    df = readdata()
    # print(df.head())

    df = datacleaning(df)
    # print(df.head())

    df = preprocessingdata(df)
    # print(df.columns)
    X = df.drop(output_column, axis=1)
    y = df[output_column]
    # get_random_grid()

    # RandomForest_model(X,y)
    # SVRModel(X,y)
    # ANNModel(X,y)



if __name__ == "__main__":
    consolelog("Running main Program")
    noname()