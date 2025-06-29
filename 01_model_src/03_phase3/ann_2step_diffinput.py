

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
from collections import defaultdict


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



input_col_today = ['Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm', 'Ngày thả', 'area', 'Tuổi tôm',
                    'Nhiệt độ', 'pH', 'Độ mặn', 'Mực nước', 'Độ trong', 'Độ kiềm']

input_col_tomorrow =   ['Nhiệt độ', 'Giống tôm', 'area', 'TDS', 'Độ cứng', 'pH', 'Loại ao', 'Công nghệ nuôi', 'Tuổi tôm', 'Độ kiềm tdpred']  # độ kiềm dự đoán sẽ được thêm sau

# predict_input_col= [
#     'Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm', 'Ngày thả', 
#     'area', 'Tuổi tôm', 'Nhiệt độ', 'pH', 'Độ mặn', 'Mực nước', 'Độ trong'
# ]

# col_need = list(set(input_col_list + predict_input_col))

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
shiftday = -3

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

    return df1


from sklearn.preprocessing import StandardScaler

def ANN_random_cv_joint(
    X_today, X_tomorrow_base, y_today, y_tomorrow,
    input_col_today, input_col_tomorrow,
    n_splits=5, test_size=0.2, random_state=42
):
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    fold = 1
    results = defaultdict(list)
    metrics_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": []}
    y_test_all = []
    y_pred_all = []

    for train_idx, test_idx in ss.split(X_today):
        log_and_flush(logger, f"Fold {fold}:")

        # === BƯỚC 1: Dự đoán độ kiềm hôm nay ===
        model_today = RandomForestRegressor(
            n_estimators=300, max_depth=20,
            min_samples_split=5, min_samples_leaf=5,
            max_features='sqrt', random_state=random_state,
            bootstrap=False, verbose=0
        )
        model_today.fit(X_today.iloc[train_idx], y_today.iloc[train_idx])
        alkaline_today_pred = model_today.predict(X_today.iloc[test_idx])

        # === BƯỚC 2: Dự đoán độ kiềm ngày mai ===
        X_tomorrow_train = X_tomorrow_base.iloc[train_idx].copy()
        X_tomorrow_test = X_tomorrow_base.iloc[test_idx].copy()
        X_tomorrow_test["Độ kiềm tdpred"] = alkaline_today_pred

        # Tạo bộ train augmented: dùng y_today (ground truth) để thêm vào feature "Độ kiềm tdpred"
        X_tomorrow_train_aug = X_tomorrow_train.assign(**{"Độ kiềm tdpred": y_today.iloc[train_idx]})

        # === Chuẩn hóa dữ liệu ===
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_tomorrow_train_aug)
        X_test_scaled = scaler_X.transform(X_tomorrow_test)

        y_train_scaled = scaler_y.fit_transform(y_tomorrow.iloc[train_idx].values.reshape(-1, 1)).flatten()

        # === Mô hình ANN ===
        inputshape = X_train_scaled.shape[1]

        model1 = Sequential()
        model1.add(Input(shape=(inputshape,)))
        model1.add(Dense(72, kernel_initializer='he_uniform', activation='relu'))
        model1.add(Dropout(0.1))
        model1.add(Dense(60, kernel_initializer='he_uniform', activation='relu'))
        model1.add(Dropout(0.1))
        model1.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
        model1.add(Dropout(0.1))
        model1.add(Dense(8, kernel_initializer='he_uniform', activation='relu'))
        model1.add(Dropout(0.1))
        model1.add(Dense(1))

        model1.compile(loss='mae', optimizer='nadam', metrics=['mae'])
        model1.fit(X_train_scaled, y_train_scaled, epochs=50, verbose=0)

        # Dự đoán và đảo ngược chuẩn hóa
        y_pred_scaled = model1.predict(X_test_scaled).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test = y_tomorrow.iloc[test_idx]

        # Lưu lại kết quả
        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)

        fold_metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
        for k, v in fold_metrics.items():
            metrics_result[k].append(v)
            results[k].append(v)
            log_and_flush(logger, f"{k}: {v:.3f}")
        
        fold += 1

    log_and_flush(logger, "----- Summary (Mean ± Std) -----")
    for metric in metrics_result:
        mean_val = np.mean(metrics_result[metric])
        std_val = np.std(metrics_result[metric])
        log_and_flush(logger, f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

    return results, y_test_all, y_pred_all






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
    log_path = os.path.join(log_dir, f"ann_2stepdiffinput_{current_time.strftime('%y%m%d-%H%M%S')}.log")
    logger = setup_logger(log_path)
    log_and_flush(logger, "ANN 2 Step (diff input)")

    df = readdata()
    df = datacleaning(df)

    input_col = list(set(input_col_today+input_col_tomorrow[:-1])) # 'Độ kiềm tdpred' sẽ đc thêm vào sau
    log_and_flush(logger,f"Step1 colums: {input_col_today}")
    log_and_flush(logger,f"Step2 colums: {input_col_tomorrow}")
    log_and_flush(logger,f"Day shift to predict: {shiftday}")

    categorical_usecol = [_col for _col in categorical_usecol_all if _col in input_col]
    print(f"{categorical_usecol=}")
    df = create_lag_features_and_target_tomorrow(df,window_size=0)
    df = preprocessingdata(df)
    log_and_flush(logger, f"DataFrame preprocess columns: {df.columns}")

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
    print(f"{X.columns=}")
    y = df[output_column]
    y_today = df['Độ kiềm']

    metrics_result, y_true, y_pred = ANN_random_cv_joint(
                    X_today, X_tomorrow_base, y_today, y_tomorrow,
                    input_col_today, input_col_tomorrow,
                    n_splits=20,
                    test_size=0.3
                )

    # metrics_result, y_true, y_pred = RandomForest_random_cv(
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