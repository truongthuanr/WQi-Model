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


input_col1 = [
    # 'Season', 
    'Loại ao', 
    'Công nghệ nuôi',  
    # 'Mực nước',
    'Tuổi tôm',
    #  'area', 
    'Nhiệt độ', 
    'pH', 
    # 'DO',
    'Độ mặn', 
    'TDS', 
    # 'Độ đục',
    'Độ trong',
    'Độ cứng',
    'Độ màu',
    ]

input_col2 = [
    # 'Season', 
    'Loại ao', 
    'Công nghệ nuôi',  
    # 'Mực nước',
    'Tuổi tôm',
    #  'area', 
    'Nhiệt độ', 
    'pH', 
    # 'DO',
    'Độ mặn', 
    # 'TDS', 
    # 'Độ đục',
    'Độ trong',
    # 'Độ cứng',
    # 'Độ màu',
    ]

input_col3 = [
    # 'Season', 
    # 'Loại ao', 
    'Công nghệ nuôi',  
    # 'Mực nước',
    'Tuổi tôm',
    #  'area', 
    'Nhiệt độ', 
    'pH', 
    # 'DO',
    'Độ mặn', 
    # 'TDS', 
    # 'Độ đục',
    'Độ trong',
    # 'Độ cứng',
    # 'Độ màu',
    ]

input_col4 = [
    # 'Season', 
    # 'Loại ao', 
    # 'Công nghệ nuôi',  
    # # 'Mực nước',
    # 'Tuổi tôm',
    # #  'area', 
    # 'Nhiệt độ', 
    # 'pH', 
    # # 'DO',
    # 'Độ mặn', 
    # 'TDS', 
    # # 'Độ đục',
    # 'Độ trong',
    # 'Độ cứng',
    # 'Độ màu',
    ]
input_col_list = [
    # ["Độ trong", "Độ màu", "TDS", "Loại ao", "Công nghệ nuôi", "Độ mặn", "Độ cứng", "Nhiệt độ", "pH", "Tuổi tôm"],
    # ["Độ màu", "TDS", "Loại ao", "Công nghệ nuôi", "Độ mặn", "Độ cứng", "Nhiệt độ", "pH", "Tuổi tôm"],
    # ["TDS", "Loại ao", "Công nghệ nuôi", "Độ mặn", "Độ cứng", "Nhiệt độ", "pH", "Tuổi tôm"],
    # ["Loại ao", "Công nghệ nuôi", "Độ mặn", "Độ cứng", "Nhiệt độ", "pH", "Tuổi tôm"],
    # ["Công nghệ nuôi", "Độ mặn", "Độ cứng", "Nhiệt độ", "pH", "Tuổi tôm"],
    # ["Độ mặn", "Độ cứng", "Nhiệt độ", "pH", "Tuổi tôm"],
    # ["Độ trong", "Loại ao", "Công nghệ nuôi", "Độ mặn", "Nhiệt độ", "pH", "Tuổi tôm"],
    # ["Season", "Loại ao", "Công nghệ nuôi", "Giống tôm", "Ngày thả", "area", "Tuổi tôm", "Nhiệt độ", "pH", "Độ mặn", "Mực nước", "Độ trong"]
    ["Công nghệ nuôi", "Giống tôm", "Tuổi tôm", "Nhiệt độ", "pH", "Độ mặn", "Mực nước", "Độ trong", "Loại ao"]

]

# input_col_list = [input_col1, input_col2,input_col3,
#                 #   input_col4 
#                   ]

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

def mean_error(y_true, y_pred):
    return np.mean(y_true - y_pred)

def svg_model(X: pd.DataFrame, y, n_repeats: int = 200, test_size: float = 0.33):
    ts_for_file = datetime.strftime(currenttime, "%y%m%d-%H%M%S")
    log_path = f"./output/svg_{ts_for_file}.log"

    metrics_result = {
        "RMSE_test": [], "RMSE_train": [],
        "MAE_test": [], "MAE_train": [],
        "MAPE_test": [], "MAPE_train": [],
        "MPE_test": [], "MPE_train": [],
        "R2_test": [], "R2_train": [],
        "ME_test": [], "ME_train": []
    }

    last_y_test_np = None
    last_y_pred_inv = None

    with open(log_path, "a+", encoding="utf-8") as logfile:
        ts_for_text = datetime.strftime(currenttime, "%y-%m-%d %H:%M:%S")
        logfile.write("SVR (RBF) - Repeated Random Splits\n")
        logfile.write(f"Time Record:\t {ts_for_text}\n")
        logfile.write(f"Input columns:\t {input_col}\n")
        logfile.write(f"Data train columns:\t {list(X.columns)}\n")
        logfile.write(f"Repeats:\t {n_repeats}\n")
        logfile.write(f"Test size:\t {test_size}\n\n")
        logfile.write(
            "RMSE_test\tRMSE_train\tMAE_test\tMAE_train\tMAPE_test\tMAPE_train\tMPE_test\tMPE_train\tR2_test\tR2_train\n"
        )

        for repeat in range(n_repeats):
            # Train/Test split (vary random_state)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=repeat
            )

            # Scaling per split
            X_sc = StandardScaler().fit(X_train)
            X_train_tf = X_sc.transform(X_train)
            y_sc = StandardScaler().fit(y_train)
            y_train_tf = y_sc.transform(y_train)

            # Model
            SVRreg = svm.SVR(kernel='rbf', epsilon=0.5, C=10)
            SVRreg.fit(X_train_tf, np.reshape(y_train_tf, (-1)))

            # Predictions (test + train)
            y_pred_test = SVRreg.predict(X_sc.transform(X_test)).reshape(-1, 1)
            y_pred_train = SVRreg.predict(X_train_tf).reshape(-1, 1)

            # Inverse transform targets
            y_test_np = y_test.to_numpy()
            y_train_np = y_train.to_numpy()
            y_pred_inv = y_sc.inverse_transform(y_pred_test)
            y_train_inv = y_sc.inverse_transform(y_pred_train)

            # Metrics
            rmse_test = root_mean_squared_error(y_test_np, y_pred_inv)
            rmse_train = root_mean_squared_error(y_train_np, y_train_inv)
            mae_test = mean_absolute_error(y_test_np, y_pred_inv)
            mae_train = mean_absolute_error(y_train_np, y_train_inv)
            mape_test = mean_absolute_percentage_error(y_test_np, y_pred_inv) * 100
            mape_train = mean_absolute_percentage_error(y_train_np, y_train_inv) * 100
            me_test = mean_error(y_test_np, y_pred_inv)
            me_train = mean_error(y_train_np, y_train_inv)

            def _mpe(y_true, y_hat):
                y_true = y_true.reshape(-1)
                y_hat = y_hat.reshape(-1)
                mask = y_true != 0
                if not np.any(mask):
                    return np.nan
                return np.mean((y_true[mask] - y_hat[mask]) / y_true[mask]) * 100

            mpe_test = _mpe(y_test_np, y_pred_inv)
            mpe_train = _mpe(y_train_np, y_train_inv)

            r2_test = r2_score(y_test_np, y_pred_inv)
            r2_train = r2_score(y_train_np, y_train_inv)

            # Accumulate
            metrics_result["RMSE_test"].append(rmse_test)
            metrics_result["RMSE_train"].append(rmse_train)
            metrics_result["MAE_test"].append(mae_test)
            metrics_result["MAE_train"].append(mae_train)
            metrics_result["MAPE_test"].append(mape_test)
            metrics_result["MAPE_train"].append(mape_train)
            metrics_result["MPE_test"].append(mpe_test)
            metrics_result["MPE_train"].append(mpe_train)
            metrics_result["R2_test"].append(r2_test)
            metrics_result["R2_train"].append(r2_train)
            metrics_result["ME_test"].append(me_test)
            metrics_result["ME_train"].append(me_train)
            
            

            logfile.write(
                f"{rmse_test:.3f}\t{rmse_train:.3f}\t{mae_test:.3f}\t{mae_train:.3f}\t{mape_test:.3f}\t{mape_train:.3f}\t{mpe_test:.3f}\t{mpe_train:.3f}\t{r2_test:.3f}\t{r2_train:.3f}\t{me_train}\t{me_test}\n"
            )

            # Keep last predictions for CSV
            last_y_test_np = y_test_np
            last_y_pred_inv = y_pred_inv

        # Summary
        logfile.write("\n----- Summary (Mean +/- Std) -----\n")
        for k in metrics_result:
            mean_val = float(np.mean(metrics_result[k]))
            std_val = float(np.std(metrics_result[k]))
            logfile.write(f"{k}: {mean_val:.3f} +/- {std_val:.3f}\n")

    # Save predictions for later use
    try:
        if last_y_test_np is not None and last_y_pred_inv is not None:
            df_save = pd.DataFrame({
                'y_test': last_y_test_np.reshape(-1),
                'y_pred': last_y_pred_inv.reshape(-1)
            })
            out_csv = os.path.join(output_folder, f"svr_last_predictions_{ts_for_file}.csv")
            df_save.to_csv(out_csv, index=False)
    except Exception as e:
        print(f"Failed to save SVR predictions CSV: {e}")
        
    

# def get_random_grid():
#     # Number of trees in random forest
#     n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
#     # Number of features to consider at every split
#     max_features = ['auto', 'sqrt']
#     # Maximum number of levels in tree
#     max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#     max_depth.append(None)
#     # Minimum number of samples required to split a node
#     min_samples_split = [2, 5, 10]
#     # Minimum number of samples required at each leaf node
#     min_samples_leaf = [1, 2, 4]
#     # Method of selecting samples for training each tree
#     bootstrap = [True, False]
#     # Create the random grid
#     random_grid = {'n_estimators': n_estimators,
#                 'max_features': max_features,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'min_samples_leaf': min_samples_leaf,
#                 'bootstrap': bootstrap}
#     # print(random_grid)
#     return random_grid


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

        svg_model(X,y)




if __name__ == "__main__":
    print("Running main Program")
    noname()
