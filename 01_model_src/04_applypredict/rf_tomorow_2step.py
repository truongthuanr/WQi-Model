
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
from typing import Tuple, List
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from typing import Tuple, Any
from collections import defaultdict
columns = [
        'Date', 
           'Season', 
           'Vụ nuôi', 
        'module_name', 
        'ao', 
           'Ngày thả', 
        'Time',
        'Nhiệt độ', 
        'pH',
         'Độ mặn', 
           'TDS',
         'Độ đục', 'DO', 
        'Độ màu',
         'Độ trong',
        'Độ kiềm', 
           'Độ cứng',
           'Loại ao',
             'Công nghệ nuôi', 
             'area', 
           'Giống tôm',
            'Tuổi tôm', 
            'Mực nước', 
        # 'Amoni', 
        #     'Nitrat', 'Nitrit', 'Silica',
            #  'Canxi', 'Kali', 'Magie'
             ]


# input_col = ["Độ màu","area","Độ mặn","Loại ao","Độ cứng","TDS","pH","Tuổi tôm", "Độ kiềm"]
input_col_today =  ['Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm', 'Ngày thả', 'area', 'Tuổi tôm',
                    'Nhiệt độ', 'pH', 'Độ mặn', 'Mực nước', 'Độ trong', 'Độ kiềm']

input_col_tomorrow = ['Độ màu','area','Độ mặn','Loại ao','Độ cứng','TDS','pH','Tuổi tôm','Độ kiềm tdpred']

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
output_column = ['Độ kiềm_tomorrow']
zscore_lim =  3
shiftday = -1
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

def create_lag_features_and_target_tomorrow(
    df: pd.DataFrame, column: str = 'Độ kiềm', window_size: int = 5
) -> pd.DataFrame:
    """
    Tạo lag features cho cột `column` (Độ kiềm) và target là giá trị của ngày hôm sau.
    Áp dụng riêng biệt cho từng 'unit', sắp xếp theo Date.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' không tồn tại trong DataFrame.")
    log_and_flush(logger, f"Tạo lag feature cho '{column}', window size = {window_size}")
    log_and_flush(logger, f"Original shape: {df.shape}")
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values(by=['units', 'Date'])
    lagged_dfs = []
    for unit, group in df_sorted.groupby('units'):
        group = group.sort_values(by='Date').reset_index(drop=True)
        # Tạo lag features
        for i in range(1, window_size + 1):
            group[f'{column}_lag_{i}'] = group[column].shift(i)
        # Tạo target là giá trị của ngày mai
        group[f'{column}_tomorrow'] = group[column].shift(shiftday)
        lagged_dfs.append(group)
    df_lagged = pd.concat(lagged_dfs, axis=0).reset_index(drop=True)
    before_drop = df_lagged.shape[0]
    df_lagged.dropna(inplace=True)
    after_drop = df_lagged.shape[0]
    log_and_flush(logger, f"Shape after dropna: {after_drop}, dropped {before_drop - after_drop} rows.")
    return df_lagged

def readdata(filepath: str) -> pd.DataFrame:
    print("Read data!")
    # df = pd.read_csv("./../../../dataset/data_4perday_cleaned.csv", usecols=columns)
    df = pd.read_csv(filepath, usecols=columns)
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
def datacleaning_val(df: pd.DataFrame) -> pd.DataFrame:
    # Cột 'Mực nước', thay các ô có giá trị = 0 thành NaN
    # Thay các giá trị NaN bằng median của cột
    df.loc[df['Mực nước']==0,'Mực nước']=np.NaN
    df['Mực nước'].fillna(df['Mực nước'].median(), inplace=True)
    # Drop cols "Time"
    df.drop(['Time'], axis=1,inplace=True)
    # Format "Date"
    df['Date'] = pd.to_datetime(df['Date'],format='%m/%d/%Y  %H:%M')
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
def preprocessingdata(df: pd.DataFrame)-> Tuple[pd.DataFrame, Any]:
    print("----- Preprocessing -----")
    df_num = df.drop(categorical_col,axis=1).copy()
    df1 = df[(np.abs(stats.zscore(df_num))<zscore_lim).all(axis=1)].copy()
    # Giữ lại tất cả cột cần thiết cho huấn luyện
    selected_cols = input_col + output_column
    # ✅ Giữ lại thêm các cột lag nếu có
    lag_cols = [col for col in df.columns if 'lag' in col]
    selected_cols += lag_cols
    df1 = df[selected_cols + categorical_usecol].copy()

    # df1 = df1[input_col + output_column].copy()
    df1.reset_index(drop=True,inplace=True)

    print("One hot encorder")
    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(df1[categorical_usecol])
    oh_df = pd.DataFrame(oh_enc.transform(df1[categorical_usecol]),
                     columns=oh_enc.get_feature_names_out()
                    )
    print(oh_df.columns)
    df1 = pd.concat([oh_df,df1],axis=1)
    df1.drop(categorical_usecol,inplace=True,axis=1)

    return df1, oh_enc


def preprocessing_testdata(df: pd.DataFrame, oh_enc)-> pd.DataFrame:
    print("----- Test Preprocessing -----")
    df_num = df.drop(categorical_col,axis=1).copy()

    df1 = df[(np.abs(stats.zscore(df_num))<zscore_lim).all(axis=1)].copy()

    # Giữ lại tất cả cột cần thiết cho huấn luyện
    selected_cols = input_col + output_column

    # ✅ Giữ lại thêm các cột lag nếu có
    lag_cols = [col for col in df.columns if 'lag' in col]
    selected_cols += lag_cols

    df1 = df[selected_cols + categorical_usecol].copy()

    # df1 = df1[input_col + output_column].copy()
    df1.reset_index(drop=True,inplace=True)

    print("One hot encorder")
    # oh_enc = OneHotEncoder(sparse_output=False)
    # oh_enc.fit(df1[categorical_usecol])
    oh_df = pd.DataFrame(oh_enc.transform(df1[categorical_usecol]),
                     columns=oh_enc.get_feature_names_out()
                    )
    print(oh_df.columns)
    df1 = pd.concat([oh_df,df1],axis=1)
    df1.drop(categorical_usecol,inplace=True,axis=1)
    return df1
def RandomForest_random_cv(
    X_today, X_tomorrow_base, y_today, y_tomorrow,
    Xval_today, Xval_tomorrow_base, yval_today, yval_tomorrow,
    n_splits: int = 10,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[dict, List[float], List[float]]:
    """
    Random Cross-Validation for Random Forest Regression (non-sliding).

    Parameters:
    - X, y: DataFrames with lag features already embedded
    - n_splits: number of cross-validation folds
    - test_size: proportion of test set (float, e.g., 0.1 for 10%)
    - random_state: random seed for reproducibility

    Returns:
    - metrics_result: dict of metrics with mean and std
    - y_test_all: list of all test targets across folds
    - y_pred_all: list of all predicted values across folds
    """
    log_and_flush(logger, "Random Forest - Random Cross-Validation")
    # log_and_flush(logger, f"Input columns: {list(X.columns)}")
    # log_and_flush(logger, f"CV Folds: {n_splits}, Test size: {test_size}")

    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    metrics_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": []}
    metrics_val_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": []}

    y_test_all = []
    y_pred_all = []
    y_val_test_all = []
    y_val_pred_all = []
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    fold = 1
    results = defaultdict(list)

    for train_idx, test_idx in ss.split(X_today):
        log_and_flush(logger, f"Fold {fold}:")
        # fold += 1

        # === BƯỚC 1: Dự đoán Độ kiềm hôm nay ===
        model_today = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=random_state,
            bootstrap=False,
            verbose=0
        )
        model_today.fit(X_today.iloc[train_idx], y_today.iloc[train_idx])
        alkaline_today_pred = model_today.predict(X_today.iloc[test_idx])

        # === BƯỚC 2: Dự đoán Độ kiềm ngày mai ===
        # Gộp dự đoán độ kiềm hôm nay vào input cho ngày mai
        X_tomorrow_train = X_tomorrow_base.iloc[train_idx].copy()
        X_tomorrow_test = X_tomorrow_base.iloc[test_idx].copy()
        X_tomorrow_test["Độ kiềm tdpred"] = alkaline_today_pred
        model_tomorrow = RandomForestRegressor(
                                n_estimators=300,
                                max_depth=20,
                                min_samples_split=5,
                                min_samples_leaf=5,
                                max_features='sqrt',
                                random_state=random_state,
                                bootstrap=False,
                                verbose=0
                            )

        model_tomorrow.fit(X_tomorrow_train.assign(**{"Độ kiềm tdpred": y_today.iloc[train_idx]}), y_tomorrow.iloc[train_idx])

        y_pred = model_tomorrow.predict(X_tomorrow_test)
        y_test = y_tomorrow.iloc[test_idx]

        # Ghi kết quả
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # Tính metrics
        fold_metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

        # Tính metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)

        metrics_result["RMSE"].append(rmse)
        metrics_result["MAE"].append(mae)
        metrics_result["MAPE"].append(mape)
        metrics_result["R2"].append(r2)

        for k, v in fold_metrics.items():
            results[k].append(v)
            log_and_flush(logger, f"{k}: {v:.3f}")
        fold += 1

    
        # start_time = time.time()
        # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # # Chuẩn hóa
        # X_scaler = StandardScaler()
        # y_scaler = StandardScaler()
        # X_train_scaled = X_scaler.fit_transform(X_train)
        # y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        # rf = RandomForestRegressor(
        #     n_estimators=300,
        #     max_depth=20,
        #     min_samples_split=5,
        #     min_samples_leaf=5,
        #     max_features='sqrt',
        #     random_state=fold,
        #     bootstrap=False,
        #     verbose=0
        # )
        # rf.fit(X_train_scaled, y_train_scaled.ravel())
        # y_pred_scaled = rf.predict(X_scaler.transform(X_test)).reshape(-1, 1)
        # y_pred = y_scaler.inverse_transform(y_pred_scaled)
        # y_true = y_test.values.reshape(-1, 1)
        # rmse = root_mean_squared_error(y_true, y_pred)
        # mae = mean_absolute_error(y_true, y_pred)
        # mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        # r2 = r2_score(y_true, y_pred)
        # metrics_result["RMSE"].append(rmse)
        # metrics_result["MAE"].append(mae)
        # metrics_result["MAPE"].append(mape)
        # metrics_result["R2"].append(r2)
        # y_test_all.extend(y_true.flatten())
        # y_pred_all.extend(y_pred.flatten())
        # log_and_flush(logger, f"[Fold {fold}] RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.3f}, R2: {r2:.3f}")
        # log_and_flush(logger, f"[Fold {fold}] Completed in {time.time() - start_time:.2f} seconds")
        #-------------------------------------------------------------------------------------------------------
        alkaline_today_pred = model_today.predict(Xval_today)
        Xval_tomorrow = Xval_tomorrow_base.copy()
        Xval_tomorrow["Độ kiềm tdpred"] = alkaline_today_pred
        yval_pred = model_tomorrow.predict(Xval_tomorrow)

        y_val_test_all.extend(yval_tomorrow)
        y_val_pred_all.extend(yval_pred)




        # y_val_pred_scaled = rf.predict(X_scaler.transform(X_val)).reshape(-1, 1)
        # y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
        yval_true = yval_tomorrow.values.reshape(-1, 1)
        rmse = root_mean_squared_error(yval_true, yval_pred)
        mae = mean_absolute_error(yval_true, yval_pred)
        mape = mean_absolute_percentage_error(yval_true, yval_pred) * 100
        r2 = r2_score(yval_true, yval_pred)
        metrics_val_result["RMSE"].append(rmse)
        metrics_val_result["MAE"].append(mae)
        metrics_val_result["MAPE"].append(mape)
        metrics_val_result["R2"].append(r2)
        # y_val_test_all.extend(y_val_true.flatten())
        # y_val_pred_all.extend(y_val_pred.flatten())
    # sample plot, just 1 fold
    plot_predictions_sorted_by_groundtruth(y_test, y_pred,
                    save_path=f"{output_folder}/predictions_groundtruth_sorted_{currenttime.strftime('%y%m%d-%H%M%S')}.png")
    
    # sample plot, just 1 fold
    plot_predictions_sorted_by_groundtruth(yval_tomorrow, yval_pred,
                    save_path=f"{output_folder}/val_groundtruth_sorted_{currenttime.strftime('%y%m%d-%H%M%S')}.png")
    
    # Summary
    log_and_flush(logger, "----- Summary (Mean ± Std) -----")
    for metric in metrics_result:
        mean_val = np.mean(metrics_result[metric])
        std_val = np.std(metrics_result[metric])
        log_and_flush(logger, f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
    log_and_flush(logger, "----- Summary Val Data (Mean ± Std) -----")
    for metric in metrics_val_result:
        mean_val = np.mean(metrics_val_result[metric])
        std_val = np.std(metrics_val_result[metric])
        log_and_flush(logger, f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
    return metrics_result, y_test_all, y_pred_all

def plot_walk_forward_metrics(metrics_result: dict, save_path: str = "walk_forward_metrics.png"):
    plt.figure(figsize=(14, 6))
    for idx, (key, values) in enumerate(metrics_result.items(), 1):
        plt.subplot(2, 2, idx)
        plt.plot(values, marker='o')
        plt.title(f"{key} per Walk-Forward Step")
        plt.xlabel("Step")
        plt.ylabel(key)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_predictions_over_time(y_true: List[float], y_pred: List[float], save_path: str = "predictions_over_time.png"):
    plt.figure(figsize=(16, 5))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title("Actual vs Predicted Values Over Time (Walk-Forward)", fontsize=14)
    plt.xlabel("Time Step")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_predictions_sorted_by_groundtruth(y_true: List[float], y_pred: List[float], 
                                           save_path: str = "predictions_sorted_by_groundtruth.png"):
    # Chuyển về numpy để dễ xử lý
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Lấy chỉ số sắp xếp theo y_true tăng dần
    sorted_idx = np.argsort(y_true)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]
    plt.figure(figsize=(12, 5))
    # plt.plot(y_true_sorted, label='Actual (Sorted)', color='blue', linewidth=2)
    plt.plot(y_true, label='Actual', color='#34495e', linewidth=1)

    # plt.plot(y_pred_sorted,  label='Predicted (Sorted by Actual)',  color='orange', linestyle='', marker='2')
    plt.plot(y_pred,  label='Predicted',  color='#3498db', linestyle='--', marker='', linewidth=1)

    plt.title("Actual vs Predicted Values Sorted by Actual", fontsize=14)
    # plt.xlabel("Index (Sorted by Actual Value)")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def testdata_prepare(oh_enc):
    _df = readdata("LocAn1high.csv")
    _df = datacleaning_val(_df)
    _df = create_lag_features_and_target_tomorrow(_df,window_size=0)
    _df = preprocessing_testdata(df=_df, oh_enc=oh_enc)
    # Xác định lại các cột tương ứng sau one-hot encode
    onehot_columns = _df.columns.tolist()
    # Tìm các cột tương ứng cho input_col_today
    X_today_cols = [col for col in onehot_columns if any(orig in col for orig in input_col_today)]

    # Tìm các cột tương ứng cho input_col_tomorrow (bỏ 'Độ kiềm tdpred')
    X_tomorrow_cols = [col for col in onehot_columns if any(orig in col for orig in input_col_tomorrow if orig != 'Độ kiềm tdpred')]

    # Gán lại X_today và X_tomorrow_base
    Xval_today = _df[X_today_cols]
    Xval_tomorrow_base = _df[X_tomorrow_cols]

    # Tách từng phần
    yval_today = _df['Độ kiềm']
    yval_tomorrow = _df['Độ kiềm_tomorrow']  # hoặc tên tương ứng
    _df.to_csv(os.path.join(output_folder,"databeforetest.csv"))

    return Xval_today, Xval_tomorrow_base, yval_today, yval_tomorrow
def noname():

    pd.options.display.float_format = '{:.6f}'.format

    global currenttime
    currenttime = datetime.now()
    global input_col 
    global categorical_usecol
    global logger 
    log_dir = "./output"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now()
    log_path = os.path.join(log_dir, f"randomforest_{current_time.strftime('%y%m%d-%H%M%S')}.log")
    logger = setup_logger(log_path)
    # log_and_flush(logger,f"Input: {input_col}")
    df = readdata(filepath="./../../../dataset/data_4perday_cleaned.csv")
    df = datacleaning(df)

    input_col = list(set(input_col_today+input_col_tomorrow[:-1])) # 'Độ kiềm tdpred' sẽ đc thêm vào sau
    log_and_flush(logger,f"Step1 colums: {input_col_today}")
    log_and_flush(logger,f"Step2 colums: {input_col_tomorrow}")
    log_and_flush(logger,f"Day shift to predict: {shiftday}")

    categorical_usecol = [_col for _col in categorical_usecol_all if _col in input_col]
    print(f"{categorical_usecol=}")
    df = create_lag_features_and_target_tomorrow(df,window_size=0)
    df, oh_enc = preprocessingdata(df)

    # Xác định lại các cột tương ứng sau one-hot encode
    onehot_columns = df.columns.tolist()

    # Tìm các cột tương ứng cho input_col_today
    X_today_cols = [col for col in onehot_columns if any(orig in col for orig in input_col_today)]
    
    # Tìm các cột tương ứng cho input_col_tomorrow (bỏ 'Độ kiềm tdpred')
    X_tomorrow_cols = [col for col in onehot_columns if any(orig in col for orig in input_col_tomorrow if orig != 'Độ kiềm tdpred')]
    # Gán lại X_today và X_tomorrow_base
    X_today = df[X_today_cols]
    X_tomorrow_base = df[X_tomorrow_cols]

    # Tách từng phần
    y_today = df['Độ kiềm']
    y_tomorrow = df['Độ kiềm_tomorrow']  # hoặc tên tương ứng


    df.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))
    

    X = df.drop(output_column, axis=1)
    print()
    print(f"{X.columns=}")
    y = df[output_column]

    Xval_today, Xval_tomorrow_base, yval_today, yval_tomorrow = testdata_prepare(oh_enc=oh_enc)
    # X_val = df_val.drop(output_column, axis=1)
    # y_val = df_val[output_column]
    metrics_result, y_true, y_pred = RandomForest_random_cv( X_today, X_tomorrow_base, y_today, y_tomorrow,
                                                            Xval_today, Xval_tomorrow_base, yval_today, yval_tomorrow,  
                                                            n_splits=20, test_size=0.3
                                                )
if __name__ == "__main__":
    print("Running main Program")
    noname()