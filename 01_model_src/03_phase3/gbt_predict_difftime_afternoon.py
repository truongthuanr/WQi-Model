

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
from sklearn.ensemble import GradientBoostingRegressor
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


# input_col_list = ['Độ màu', 'area', 'Độ mặn', 'Loại ao', 'Độ cứng', 'TDS', 'pH', 'Tuổi tôm', 'Độ kiềm']

input_col_list = ["Season","Loại ao","Công nghệ nuôi","Giống tôm","Ngày thả","area","Tuổi tôm",
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


output_column = ['Độ kiềm_afternoon']
zscore_lim =  3

def setup_logger(log_path):
    logger = logging.getLogger('TimeSeriesGBT')
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
    # keep 'Time' for time-of-day bucketing
    # df.drop(['Time'], axis=1,inplace=True)

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


def create_morning_noon_same_day_dataset(
    df: pd.DataFrame,
    column: str = 'Độ kiềm'
) -> pd.DataFrame:
    """
    Build dataset to predict same-day afternoon alkalinity from morning records.
    Time bucketing from 'Time':
      - hour < 9  -> 'morning'
      - 9 <= hour < 15 -> 'noon'
      - hour >= 15 -> 'afternoon'
    Keep only days with both morning and afternoon; use the last record within each bucket.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    work = df.copy()
    work['Date'] = pd.to_datetime(work['Date'])
    # Parse 'Time' that may be numeric hours like 8.0 or 11.5 (11:30)
    def _parse_time_to_hm(v):
        if pd.isna(v):
            return (np.nan, np.nan)
        s = str(v).strip()
        try:
            f = float(s)
            h = int(np.floor(f))
            m = int(round((f - h) * 60))
            if m == 60:
                h += 1
                m = 0
            return (h, m)
        except Exception:
            pass
        # try HH:MM format
        if ':' in s:
            parts = s.split(':')
            try:
                h = int(parts[0])
                m = int(float(parts[1]))
                return (h, m)
            except Exception:
                return (np.nan, np.nan)
        return (np.nan, np.nan)

    hm = work['Time'].apply(_parse_time_to_hm)
    work['_hour'] = hm.apply(lambda x: x[0])
    work['_minute'] = hm.apply(lambda x: x[1])
    work['_dt'] = work['Date'] + pd.to_timedelta(work['_hour'].fillna(0), unit='h') \
                   + pd.to_timedelta(work['_minute'].fillna(0), unit='m')

    def _bucket(h):
        if pd.isna(h):
            return np.nan
        if h < 9:
            return 'morning'
        elif 9 <= h <= 15:
            return 'noon'
        elif h > 15:
            return 'afternoon'
        else:
            return np.nan

    work['time_bucket'] = work['_hour'].apply(_bucket)
    work = work[work['time_bucket'].isin(['morning', 'noon', "afternoon"])].copy()

    work.sort_values(['units', 'Date', '_dt'], inplace=True)
    morning = work[work['time_bucket'] == 'morning'].groupby(['units', 'Date'], as_index=False).tail(1)
    afternoon = work[work['time_bucket'] == 'afternoon'].groupby(['units', 'Date'], as_index=False).tail(1)

    afternoon_target_col = f"{column}_afternoon"
    afternoon_keep = afternoon[['units', 'Date', column]].rename(columns={column: afternoon_target_col})

    base_cols = ['units', 'Date']
    feat_cols = [c for c in input_col_list if c in morning.columns]
    cat_cols = [c for c in categorical_usecol_all if c in morning.columns]
    morning_keep = morning[base_cols + feat_cols + cat_cols].copy()

    merged = pd.merge(morning_keep, afternoon_keep, on=['units', 'Date'], how='inner')

    # clean temp columns if any accidentally carried over (shouldn't be)
    for col in ['_dt', '_hour', 'time_bucket', 'Time']:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    if 'logger' in globals():
        log_and_flush(logger, f"Merged morning→ afternoon dataset shape: {merged.shape}")

    before = len(merged)
    merged.dropna(axis=0, inplace=True)
    after = len(merged)
    if 'logger' in globals():
        log_and_flush(logger, f"After dropna: {after} rows, removed {before - after}")

    return merged

def preprocessingdata(df: pd.DataFrame)-> pd.DataFrame:
    log_and_flush(logger, "----- Preprocessing -----")
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

    log_and_flush(logger, "One hot encoder")
    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(df1[categorical_usecol])
    oh_df = pd.DataFrame(oh_enc.transform(df1[categorical_usecol]),
                     columns=oh_enc.get_feature_names_out()
                    )
    try:
        log_and_flush(logger, f"OHE columns: {list(oh_df.columns)}")
    except Exception:
        # Fallback to avoid console encoding errors
        pass
    df1 = pd.concat([oh_df,df1],axis=1)
    df1.drop(categorical_usecol,inplace=True,axis=1)

    return df1


def GBT_random_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 10,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[dict, List[float], List[float]]:
    """
    Random Cross-Validation for Gradient Boosted Tree (non-sliding).

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
    log_and_flush(logger, "Gradient Boosted Tree - Random Cross-Validation")
    log_and_flush(logger, f"Input columns: {list(X.columns)}")
    log_and_flush(logger, f"CV Folds: {n_splits}, Test size: {test_size}")

    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    metrics_result = {"RMSE": [], "MAE": [], "MAPE": [], "MPE": [], "ME": [], "R2": []}
    y_test_all = []
    y_pred_all = []

    fold = 0
    for train_index, test_index in splitter.split(X):
        fold += 1
        start_time = time.time()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Chuẩn hóa
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

        gbr = GradientBoostingRegressor(n_estimators=100,
                                max_depth=10,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                max_features='sqrt', 
                                loss='squared_error',
                                random_state=0,
                                learning_rate=0.02,
                                verbose=0,
                                )

        gbr.fit(X_train_scaled, y_train_scaled.ravel())

        y_pred_scaled = gbr.predict(X_scaler.transform(X_test)).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_test.values.reshape(-1, 1)

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        # ME: mean error, MPE: mean percentage error (%)
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        err = y_pred_f - y_true_f
        me = float(np.mean(err))
        denom = np.where(np.abs(y_true_f) > 1e-8, y_true_f, 1e-8)
        mpe = float(np.mean(err / denom) * 100.0)

        metrics_result["RMSE"].append(rmse)
        metrics_result["MAE"].append(mae)
        metrics_result["MAPE"].append(mape)
        metrics_result["MPE"].append(mpe)
        metrics_result["ME"].append(me)
        metrics_result["R2"].append(r2)

        y_test_all.extend(y_true.flatten())
        y_pred_all.extend(y_pred.flatten())

        log_and_flush(logger, f"[Fold {fold}] RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.3f}, MPE: {mpe:.3f}, ME: {me:.3f}, R2: {r2:.3f}")
        log_and_flush(logger, f"[Fold {fold}] Completed in {time.time() - start_time:.2f} seconds")
    # sample plot, just 1 fold
    plot_predictions_sorted_by_groundtruth(y_true.flatten(), y_pred.flatten(),
                    save_path=f"{output_folder}/predictions_groundtruth_sorted_{currenttime.strftime('%y%m%d-%H%M%S')}.png")
    # Summary
    log_and_flush(logger, "----- Summary (Mean ± Std) -----")
    for metric in metrics_result:
        mean_val = np.mean(metrics_result[metric])
        std_val = np.std(metrics_result[metric])
        log_and_flush(logger, f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

    return metrics_result, y_test_all, y_pred_all



def plot_walk_forward_metrics(metrics_result: dict, save_path: str = "walk_forward_metrics.png"):
    n = len(metrics_result)
    if n == 0:
        return
    # dynamic grid size
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(5 * cols, 3.5 * rows))
    for idx, (key, values) in enumerate(metrics_result.items(), 1):
        plt.subplot(rows, cols, idx)
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



def plot_predictions_sorted_by_groundtruth(y_true: List[float], 
                                           y_pred: List[float], 
                                           save_path: str = "predictions_sorted_by_groundtruth.png"):
    # Chuyển về numpy để dễ xử lý
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Lấy chỉ số sắp xếp theo y_true tăng dần
    sorted_idx = np.argsort(y_true)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_sorted, label='Actual (Sorted)', color='blue', linewidth=2)
    plt.plot(y_pred_sorted, 
             label='Predicted (Sorted by Actual)', 
             color='orange', 
             linestyle='',
             marker='2')
    plt.title("Actual vs Predicted Values Sorted by Actual", fontsize=14)
    plt.xlabel("Index (Sorted by Actual Value)")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def noname():

    # np.set_printoptions(suppress=True)
    pd.options.display.float_format = '{:.6f}'.format

    # print(df.head())
    global currenttime
    currenttime = datetime.now()
    global input_col 
    global categorical_usecol
    global logger 
    log_dir = "./output"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now()
    log_path = os.path.join(log_dir, f"GBT2step_{current_time.strftime('%y%m%d-%H%M%S')}.log")
    logger = setup_logger(log_path)
    # for _input_col in input_col_list:

    # log_and_flush(logger,f"Input: {_input_col}")
    df = readdata()
    df = datacleaning(df)
    # input_col = col_need
    input_col = input_col_list
    categorical_usecol = [_col for _col in categorical_usecol_all if _col in input_col]
    try:
        log_and_flush(logger, f"categorical_usecol={categorical_usecol}")
    except Exception:
        pass
    # Override target to afternoon alkalinity from morning features
    global output_column
    output_column = ['Độ kiềm_afternoon']
    # Build morning→ afternoon same-day dataset
    df = create_morning_noon_same_day_dataset(df)
    df = preprocessingdata(df)
    
    df.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

    X = df.drop(output_column, axis=1)
    try:
        log_and_flush(logger, f"X columns={list(X.columns)}")
    except Exception:
        pass
    y = df[output_column]
    y_today = df['Độ kiềm']

    metrics_result, y_true, y_pred = GBT_random_cv(X, y, n_splits=100)

    # metrics_result, y_true, y_pred = GBT_random_cv(
    # X, y,
    # n_splits=50,
    # test_size=0.3
    # )
    log_and_flush(logger, f"Plotting..., time subfix: {currenttime.strftime('%y%m%d-%H%M%S')}")

    plot_walk_forward_metrics(metrics_result, save_path=f"{output_folder}/metrics_{currenttime.strftime('%y%m%d-%H%M%S')}.png")
    plot_predictions_over_time(y_true, y_pred,save_path=f"{output_folder}/predictions_over_time_{currenttime.strftime('%y%m%d-%H%M%S')}.png")



if __name__ == "__main__":
    print("Running main Program")
    noname()
