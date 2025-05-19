

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,\
                                  StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score,\
                            mean_squared_error,\
                            root_mean_squared_error,\
                            mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,\
                            GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.linear_model import SGDRegressor, LinearRegression
from keras.models import Sequential
from sklearn import svm
from scikeras.wrappers import KerasRegressor
import matplotlib
import time

from sklearn.utils import shuffle


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
    ["Công nghệ nuôi", "Nhiệt độ", "Độ màu", "area", "Độ mặn", "Loại ao", "Độ cứng", "TDS", "pH", "Tuổi tôm"],
    # ["Nhiệt độ", "Độ màu", "area", "Độ mặn", "Loại ao", "Độ cứng", "TDS", "pH", "Tuổi tôm"],
    # ["Độ màu", "area", "Độ mặn", "Loại ao", "Độ cứng", "TDS", "pH", "Tuổi tôm"],
    # ["area", "Độ mặn", "Loại ao", "Độ cứng", "TDS", "pH", "Tuổi tôm"],
    # ["Độ mặn", "Loại ao", "Độ cứng", "TDS", "pH", "Tuổi tôm"],
    # ["Loại ao", "Độ cứng", "TDS", "pH", "Tuổi tôm"],
    # ["Công nghệ nuôi", "Nhiệt độ", "area", "Độ mặn", "Loại ao", "pH", "Tuổi tôm"],
    # ["Season", "Loại ao", "Công nghệ nuôi", "Giống tôm", "Ngày thả", "area", "Tuổi tôm", "Nhiệt độ", "pH", "Độ mặn", "Mực nước", "Độ trong"]
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


output_column = ['Độ kiềm']
zscore_lim =  3

def setup_logger(log_path):
    logger = logging.getLogger('TimeSeriesRF')
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger

def log_and_flush(logger, msg):
    logger.info(msg)
    for handler in logger.handlers:
        handler.flush()
    
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

def create_sliding_window_by_unit(df: pd.DataFrame, column: str = 'Độ kiềm', window_size: int = 5) -> pd.DataFrame:
    """
    Tạo lag features cho cột column với window_size, áp dụng riêng biệt cho từng 'unit'
    """
    df_sorted = df.sort_values(by=['units', 'Date']).copy()
    lagged_dfs = []

    for unit, group in df_sorted.groupby('units'):
        group_lagged = group.copy()
        for i in range(1, window_size + 1):
            group_lagged[f'{column}_lag_{i}'] = group_lagged[column].shift(i)
        lagged_dfs.append(group_lagged)

    df_lagged = pd.concat(lagged_dfs, axis=0)
    df_lagged.dropna(inplace=True)
    return df_lagged


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
    print("----- Preprocessing -----")
    df_num = df.drop(categorical_col,axis=1).copy()

    df1 = df[(np.abs(stats.zscore(df_num))<zscore_lim).all(axis=1)].copy()
    print(f"{df1.columns=}")
    # fig= plt.figure(figsize=(10,5))
    # ax = sns.boxplot(df1)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    # plt.show()
    # plt.savefig(os.path.join(output_folder,"boxplot1.png"))

    df1 = df1[input_col + output_column].copy()
    df1.reset_index(drop=True,inplace=True)

    # df1.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

    # print("Plot data!")
    # plt.figure(figsize=(10,5))
    # sns.boxenplot(df1)
    # ax = plt.gca()
    # ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
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


def RandomForest_time_series(X: pd.DataFrame, y: pd.DataFrame, n_repeats: int = 10, test_size: float = 0.30):
    currenttime = datetime.now()
    _currenttime = currenttime.strftime("%y%m%d-%H%M%S")
    log_path = f"./output/randomforest_timeseries_{_currenttime}.log"

    os.makedirs("./output", exist_ok=True)

    logger = setup_logger(log_path)
    log_and_flush(logger, "Random Forest - Time Series Forecasting")
    log_and_flush(logger, f"Time Record: {currenttime.strftime('%y-%m-%d %H:%M:%S')}")
    log_and_flush(logger, f"Input columns: {list(X.columns)}")
    log_and_flush(logger, f"Repeats: {n_repeats}")
    log_and_flush(logger, f"Test size: {test_size}")
    log_and_flush(logger, f"Data length: {len(X)}")

    log_and_flush(logger, "RMSE\tMAE\tMAPE\tR2")

    metrics_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": []}

    for repeat in range(n_repeats):
        start_time = time.time()

        train_size = int(len(X) * (1 - test_size))
        offset = repeat * int(len(X) * 0.05)  # Sliding window offset
        log_and_flush(logger, f"Offset repeat: {offset}")



        if offset + train_size + 1 >= len(X):
            log_and_flush(logger, f"Break loop at repeat {repeat} because offset exceeded data length")
            break

        X_train = X.iloc[offset:offset + train_size]
        y_train = y.iloc[offset:offset + train_size]
        X_test = X.iloc[offset + train_size:]
        y_test = y.iloc[offset + train_size:]

        # Chuẩn hóa
        X_sc = StandardScaler()
        y_sc = StandardScaler()
        X_train_tf = X_sc.fit_transform(X_train)
        y_train_tf = y_sc.fit_transform(y_train.values.reshape(-1, 1))

        rf = RandomForestRegressor(
            n_estimators=800,
            max_depth=50,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=repeat,
            bootstrap=False,
            verbose=0
        )

        rf.fit(X_train_tf, y_train_tf.ravel())
        y_pred = rf.predict(X_sc.transform(X_test)).reshape(-1, 1)
        y_pred_inv = y_sc.inverse_transform(y_pred)
        y_test_np = y_test.values.reshape(-1, 1)

        # Tính các metric (tự định nghĩa hoặc import hàm root_mean_squared_error)
        rmse = root_mean_squared_error(y_test_np, y_pred_inv)
        mae = mean_absolute_error(y_test_np, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_np, y_pred_inv) * 100
        r2 = r2_score(y_test_np, y_pred_inv)

        metrics_result["RMSE"].append(rmse)
        metrics_result["MAE"].append(mae)
        metrics_result["MAPE"].append(mape)
        metrics_result["R2"].append(r2)

        log_and_flush(logger, f"{rmse:.3f}\t{mae:.3f}\t{mape:.3f}\t{r2:.3f}")

        elapsed = time.time() - start_time
        log_and_flush(logger, f"Repeat {repeat} done in {elapsed:.2f} seconds")

    # Summary
    log_and_flush(logger, "----- Summary (Mean ± Std) -----")
    for k in metrics_result:
        mean_val = np.mean(metrics_result[k])
        std_val = np.std(metrics_result[k])
        log_and_flush(logger, f"{k}: {mean_val:.3f} ± {std_val:.3f}")


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
        df = create_sliding_window_by_unit(df)
        df = preprocessingdata(df)
        df.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

        X = df.drop(output_column, axis=1)
        print()
        print(f"{X.columns=}")
        y = df[output_column]


        RandomForest_time_series(X, y, n_repeats=200)



if __name__ == "__main__":
    print("Running main Program")
    noname()