
import os
import sys
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
        #  'Độ đục', 'DO', 
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
input_col = ["Season","Loại ao","Công nghệ nuôi","Giống tôm","Ngày thả","area","Tuổi tôm",
             "Độ mặn","Nhiệt độ","pH", "Độ kiềm","Mực nước","Độ trong",]


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
# shiftday = -3

shiftday = int(sys.argv[1])
def setup_logger(log_path):
    logger = logging.getLogger('TimeSeriesSVR')
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

def mean_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        percentage_errors = np.where(y_true != 0, (y_true - y_pred) / y_true, np.nan)
    mpe = np.nanmean(percentage_errors) * 100
    if np.isnan(mpe):
        return 0.0
    return float(mpe)
def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(y_pred - y_true))

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
def SVR_random_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    n_splits: int = 10,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[dict, List[float], List[float]]:
    """
    Random Cross-Validation for SVR Regression (non-sliding).
    """
    log_and_flush(logger, "SVR - Random Cross-Validation")
    log_and_flush(logger, f"Input columns: {list(X.columns)}")
    log_and_flush(logger, f"CV Folds: {n_splits}, Test size: {test_size}")

    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    metrics_train_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": [], "MPE": [], "ME": []}
    metrics_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": [], "MPE": [], "ME": []}
    metrics_val_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": [], "MPE": [], "ME": []}

    y_test_all = []
    y_pred_all = []
    y_val_test_all = []
    y_val_pred_all = []
    train_records = []
    test_records = []
    val_records = []

    fold = 0
    for train_index, test_index in splitter.split(X):
        fold += 1
        log_and_flush(logger, f"Fold {fold}:")
        start_time = time.time()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

        svr = svm.SVR(kernel='rbf', C=100.0, gamma='scale', epsilon=0.1)
        svr.fit(X_train_scaled, y_train_scaled.ravel())

        y_train_pred_scaled = svr.predict(X_train_scaled).reshape(-1, 1)
        y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
        y_train_true = y_train.values.reshape(-1, 1)
        y_train_true_flat = y_train_true.flatten()
        y_train_pred_flat = y_train_pred.flatten()

        train_rmse = root_mean_squared_error(y_train_true_flat, y_train_pred_flat)
        train_mae = mean_absolute_error(y_train_true_flat, y_train_pred_flat)
        train_mape = mean_absolute_percentage_error(y_train_true_flat, y_train_pred_flat) * 100
        train_r2 = r2_score(y_train_true_flat, y_train_pred_flat)
        train_mpe = mean_percentage_error(y_train_true_flat, y_train_pred_flat)
        train_me = mean_error(y_train_true_flat, y_train_pred_flat)

        metrics_train_result["RMSE"].append(train_rmse)
        metrics_train_result["MAE"].append(train_mae)
        metrics_train_result["MAPE"].append(train_mape)
        metrics_train_result["R2"].append(train_r2)
        metrics_train_result["MPE"].append(train_mpe)
        metrics_train_result["ME"].append(train_me)
        log_and_flush(
            logger,
            f"[Train] RMSE: {train_rmse:6.3f}, MAE: {train_mae:6.3f}, "
            f"MAPE: {train_mape:6.3f}, R2: {train_r2:6.3f}, MPE: {train_mpe:6.3f}, ME: {train_me:6.3f}"
        )
        for t, p in zip(y_train_true_flat, y_train_pred_flat):
            train_records.append({"fold": fold, "dataset": "train", "y_true": float(t), "y_pred": float(p)})

        y_pred_scaled = svr.predict(X_scaler.transform(X_test)).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_test.values.reshape(-1, 1)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        rmse = root_mean_squared_error(y_true_flat, y_pred_flat)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat) * 100
        r2 = r2_score(y_true_flat, y_pred_flat)
        mpe = mean_percentage_error(y_true_flat, y_pred_flat)
        me = mean_error(y_true_flat, y_pred_flat)

        metrics_result["RMSE"].append(rmse)
        metrics_result["MAE"].append(mae)
        metrics_result["MAPE"].append(mape)
        metrics_result["R2"].append(r2)
        metrics_result["MPE"].append(mpe)
        metrics_result["ME"].append(me)
        y_test_all.extend(y_true_flat)
        y_pred_all.extend(y_pred_flat)
        log_and_flush(
            logger,
            f"[Test]  RMSE: {rmse:6.3f}, MAE: {mae:6.3f}, "
            f"MAPE: {mape:6.3f}, R2: {r2:6.3f}, MPE: {mpe:6.3f}, ME: {me:6.3f}"
        )
        for t, p in zip(y_true_flat, y_pred_flat):
            test_records.append({"fold": fold, "dataset": "test", "y_true": float(t), "y_pred": float(p)})

        y_val_pred_scaled = svr.predict(X_scaler.transform(X_val)).reshape(-1, 1)
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled)
        y_val_true = y_val.values.reshape(-1, 1)
        y_val_true_flat = y_val_true.flatten()
        y_val_pred_flat = y_val_pred.flatten()

        rmse = root_mean_squared_error(y_val_true_flat, y_val_pred_flat)
        mae = mean_absolute_error(y_val_true_flat, y_val_pred_flat)
        mape = mean_absolute_percentage_error(y_val_true_flat, y_val_pred_flat) * 100
        r2 = r2_score(y_val_true_flat, y_val_pred_flat)
        mpe = mean_percentage_error(y_val_true_flat, y_val_pred_flat)
        me = mean_error(y_val_true_flat, y_val_pred_flat)

        metrics_val_result["RMSE"].append(rmse)
        metrics_val_result["MAE"].append(mae)
        metrics_val_result["MAPE"].append(mape)
        metrics_val_result["R2"].append(r2)
        metrics_val_result["MPE"].append(mpe)
        metrics_val_result["ME"].append(me)
        y_val_test_all.extend(y_val_true_flat)
        y_val_pred_all.extend(y_val_pred_flat)
        log_and_flush(
            logger,
            f"[Val]   RMSE: {rmse:6.3f}, MAE: {mae:6.3f}, "
            f"MAPE: {mape:6.3f}, R2: {r2:6.3f}, MPE: {mpe:6.3f}, ME: {me:6.3f}"
        )
        for t, p in zip(y_val_true_flat, y_val_pred_flat):
            val_records.append({"fold": fold, "dataset": "validation", "y_true": float(t), "y_pred": float(p)})

    plot_predictions_sorted_by_groundtruth(
        y_true_flat,
        y_pred_flat,
        save_path=f"{output_folder}/predictions_groundtruth_sorted_{currenttime.strftime('%y%m%d-%H%M%S')}.png"
    )
    plot_predictions_sorted_by_groundtruth(
        y_val_true_flat,
        y_val_pred_flat,
        save_path=f"{output_folder}/val_groundtruth_sorted_{currenttime.strftime('%y%m%d-%H%M%S')}.png"
    )

    log_and_flush(logger, "----- Metrics Summary (Mean +/- Std) -----")
    for label, metrics_dict in [("Train", metrics_train_result), ("Test", metrics_result), ("Validation", metrics_val_result)]:
        log_and_flush(logger, f"{label} data:")
        for metric, values in metrics_dict.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            log_and_flush(logger, f"{metric}: {mean_val:.3f} +/- {std_val:.3f}")

    prediction_records = train_records + test_records + val_records
    if prediction_records:
        predictions_df = pd.DataFrame(prediction_records)
        predictions_path = os.path.join(
            output_folder,
            f"svr_random_cv_predictions_{currenttime.strftime('%y%m%d-%H%M%S')}.csv",
        )
        predictions_df.to_csv(predictions_path, index=False)
        log_and_flush(logger, f"Saved predictions snapshot to {predictions_path}")

    return metrics_result, y_test_all, y_pred_all


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
    plt.plot(y_true, label='Độ kiềm thực', color='#34495e', linewidth=1)

    # plt.plot(y_pred_sorted,  label='Predicted (Sorted by Actual)',  color='orange', linestyle='', marker='2')
    plt.plot(y_pred,  label='Độ kiềm dự đoán',  color='#3498db', linestyle='--', marker='', linewidth=1)

    # plt.title("Actual vs Predicted Values Sorted by Actual", fontsize=14)
    plt.xlabel("Số mẫu")
    plt.ylabel("Độ kiềm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def testdata_prepare(oh_enc):
    filepath = "LocAn1.csv"
    log_and_flush(logger,f"Đọc dữ liệu test từ {filepath}")
    _df = readdata(filepath)
    _df = datacleaning_val(_df)
    _df = create_lag_features_and_target_tomorrow(_df,window_size=0)
    _df = preprocessing_testdata(df=_df, oh_enc=oh_enc)
    _df.to_csv(os.path.join(output_folder,"databeforetest.csv"))
    return _df
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
    log_path = os.path.join(log_dir, f"svr_{current_time.strftime('%y%m%d-%H%M%S')}.log")
    logger = setup_logger(log_path)
    log_and_flush(logger,"------------------------------------------------------------------")
    log_and_flush(logger,f"Random Forest Model (Kiềm thực) - Dự đoán {int(shiftday*-1)} ngày tiếp theo")
    log_and_flush(logger,"------------------------------------------------------------------")
    log_and_flush(logger,f"Input: {input_col}")
    df = readdata(filepath="./../../../dataset/data_4perday_cleaned.csv")
    df = datacleaning(df)
    categorical_usecol = [_col for _col in categorical_usecol_all if _col in input_col]
    print(f"{categorical_usecol=}")
    df = create_lag_features_and_target_tomorrow(df,window_size=0)
    df, oh_enc = preprocessingdata(df)
    df_val = testdata_prepare(oh_enc=oh_enc)
    df.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))
    X = df.drop(output_column, axis=1)
    print()
    print(f"{X.columns=}")
    y = df[output_column]

    X_val = df_val.drop(output_column, axis=1)
    y_val = df_val[output_column]
    metrics_result, y_true, y_pred = SVR_random_cv( X, y, X_val,y_val,  n_splits=200, test_size=0.3
                                                )
if __name__ == "__main__":
    print("Running main Program")
    noname()
