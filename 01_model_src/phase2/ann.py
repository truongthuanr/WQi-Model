# from helper import consolelog
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.linear_model import SGDRegressor, LinearRegression
from keras.models import Sequential
from sklearn import svm
from scikeras.wrappers import KerasRegressor
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
input_col_list = [
    ["DO", "Nhiệt độ", "Giống tôm", "area", "TDS", "Độ cứng", "pH", "Loại ao", "Công nghệ nuôi", "Tuổi tôm"],
    ["Nhiệt độ", "Giống tôm", "area", "TDS", "Độ cứng", "pH", "Loại ao", "Công nghệ nuôi", "Tuổi tôm"],
    ["Giống tôm", "area", "TDS", "Độ cứng", "pH", "Loại ao", "Công nghệ nuôi", "Tuổi tôm"],
    ["area", "TDS", "Độ cứng", "pH", "Loại ao", "Công nghệ nuôi", "Tuổi tôm"],
    ["TDS", "Độ cứng", "pH", "Loại ao", "Công nghệ nuôi", "Tuổi tôm"],
    ["Độ cứng", "pH", "Loại ao", "Công nghệ nuôi", "Tuổi tôm"],
    ["DO", "Nhiệt độ", "Giống tôm", "area", "pH", "Loại ao", "Công nghệ nuôi", "Tuổi tôm"],
    ["Season", "Loại ao", "Công nghệ nuôi", "Giống tôm", "Ngày thả", "area", "Tuổi tôm", "Nhiệt độ", "pH", "Độ mặn", "Mực nước", "Độ trong"]
]
output_folder = "output"
categorical_col = ['Date',
                   'Season', 
                   'Loại ao', 
                   'Công nghệ nuôi', 
                   'Giống tôm',
                   'units']
categorical_usecol_all = [
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
        
    fig.suptitle("ANN")       
    plt.tight_layout()
    # plt.show()
    suffix = datetime.strftime(datetime.now(),"%y%m%d-%H%M%S")
    plt.savefig(os.path.join(output_folder,f"ann_{suffix}.png"))
    
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
    print("Read data!")
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
    # plt.savefig(os.path.join(output_folder,"boxplot1.png"))

    df1 = df1[input_col + output_column].copy()
    df1.reset_index(drop=True,inplace=True)

    df1.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

    print("Plot data!")
    plt.figure(figsize=(10,5))
    sns.boxenplot(df1)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    # plt.show()
    # plt.savefig(os.path.join(output_folder,"boxplot2.png"))


    print("One hot encorder")
    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(df1[categorical_usecol])
    oh_df = pd.DataFrame(oh_enc.transform(df1[categorical_usecol]),
                     columns=oh_enc.get_feature_names_out()
                    )
    print(oh_df.columns)
    df1 = pd.concat([oh_df,df1],axis=1)
    df1.drop(categorical_usecol,inplace=True,axis=1)

    return df1


def ANN_model_with_repeated_random_subsampling(X: pd.DataFrame, y: pd.DataFrame, n_repeats: int = 10, test_size: float = 0.33):
    _currenttime = datetime.strftime(currenttime, "%y%m%d-%H%M%S")
    log_path = f"./output/ann_repeated_{_currenttime}.log"

    metrics_result = {
        "RMSE": [], "MAE": [], "MAPE": [], "R2": []
    }

    with open(log_path, "a+", encoding="utf-8") as logfile:
        logfile.write("ANN - Repeated Random Splits\n")
        logfile.write(f"Time Record:\t {datetime.strftime(currenttime, '%y-%m-%d %H:%M:%S')}\n")
        logfile.write(f"Input columns:\t {input_col}\n")
        logfile.write(f"Repeats:\t {n_repeats}\n")
        logfile.write(f"Test size:\t {test_size}\n\n")
        logfile.write("RMSE\tMAE\tMAPE\tR2\n")

        for repeat in range(n_repeats):
            print(f"----- SET {repeat} -----")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=repeat
            )

            X_sc = StandardScaler()
            y_sc = StandardScaler()
            X_train_tf = X_sc.fit_transform(X_train)
            y_train_tf = y_sc.fit_transform(y_train)

            #     # define the model
            model1 = Sequential()
            inputshape = len(X.columns)
            model1.add(Input(shape=(inputshape,)))
            # Layer #
            model1.add(Dense(72,kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(0.1))
            # # Layer #
            model1.add(Dense(60,kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(0.1))
            # # Layer #
            model1.add(Dense(32,kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(0.1))

            # Layer #
            model1.add(Dense(8, kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(0.1))

            model1.add(Dense(1))
            model1.compile(loss='mae', 
                        optimizer='nadam',
                        metrics=['accuracy','r2_score']
                        )
            # model1.summary()


            # Train model
            history = model1.fit(X_train_tf, y_train_tf,
                                epochs=300,
                                batch_size=32,
                                verbose=False,
                                validation_split=0.2)
            
            # Evaluate model on the test set
            y_pred = model1.predict(X_sc.transform(X_test))
            y_pred = y_sc.inverse_transform(y_pred)
            
            # Calculate errors
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            r2 = r2_score(y_test, y_pred)

            metrics_result["RMSE"].append(rmse)
            metrics_result["MAE"].append(mae)
            metrics_result["MAPE"].append(mape)
            metrics_result["R2"].append(r2)

            logfile.write(f"{rmse:.3f}\t{mae:.3f}\t{mape:.3f}\t{r2:.3f}\n")

        logfile.write("\n----- Summary (Mean ± Std) -----\n")
        for k in metrics_result:
            mean_val = np.mean(metrics_result[k])
            std_val = np.std(metrics_result[k])
            logfile.write(f"{k}: {mean_val:.3f} ± {std_val:.3f}\n")
        

def noname():
    # np.set_printoptions(suppress=True)
    pd.options.display.float_format = '{:.6f}'.format

    # print(df.head())
    global currenttime
    currenttime = datetime.now()
    global input_col 
    global categorical_usecol
    for _input_col in input_col_list:
        df = readdata()
        df = datacleaning(df)
        input_col = _input_col
        categorical_usecol = [_col for _col in categorical_usecol_all if _col in _input_col]
        print(f"{categorical_usecol=}")
        df = preprocessingdata(df)
        # print(df.columns)
        X = df.drop(output_column, axis=1)
        print()
        print(f"{X.columns=}")
        print(f"{len(X.columns)=}")

        y = df[output_column]
        # get_random_grid()

        ANN_model_with_repeated_random_subsampling(X, y, n_repeats=200)



if __name__ == "__main__":
    print("Running main Program")
    noname()
