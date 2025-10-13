import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List


# Danh sách cột chuẩn tiếng Việt (tham khảo)
columns = ['Date',
           'Season',
           'Vụ nuôi', 'module_name', 'ao',
           'Ngày thả', 'Time', 'Nhiệt độ', 'pH', 'Độ mặn',
           'TDS', 'Độ đục', 'DO', 'Độ màu', 'Độ trong', 'Độ kiềm',
           'Độ cứng', 'Loại ao', 'Công nghệ nuôi', 'area',
           'Giống tôm', 'Tuổi tôm', 'Mực nước', 'Amoni',
           'Nitrat', 'Nitrit', 'Silica']

input_col_list = [
    'Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm', 'Ngày thả',
    'area', 'Tuổi tôm',
    'Độ mặn', 'Nhiệt độ', 'pH', 'Độ kiềm', 'Mực nước', 'Độ trong'
]

categorical_col = ['Date', 'Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm', 'units']
categorical_usecol_all = ['Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm']

output_folder = 'output'
output_column = ['Độ kiềm_afternoon']
zscore_lim = 3


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('TimeSeriesRF')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


def log_and_flush(logger: logging.Logger, msg: str):
    logger.info(msg)
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def readdata() -> pd.DataFrame:
    print('Read data!')
    # Đọc full, tránh lỗi do khác tên cột
    df = pd.read_csv('./../../../dataset/data_4perday_cleaned.csv')
    # Chuẩn hóa tên cột hay gặp
    if 'Amoni' in df.columns:
        df.rename({'Amoni': 'TAN'}, axis=1, inplace=True)
    return df


def datacleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Chuẩn hóa dữ liệu cơ bản
    if 'Mực nước' in df.columns:
        df.loc[df['Mực nước'] == 0, 'Mực nước'] = np.NaN
        df['Mực nước'].fillna(df['Mực nước'].median(), inplace=True)

    # Định dạng cột ngày
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Chuyển 'Tuổi tôm' sang số nếu có
    if 'Tuổi tôm' in df.columns:
        df['Tuổi tôm'] = df['Tuổi tôm'].apply(
            lambda x: int(float(x)) if str(x).replace('.', '', 1).isnumeric() else np.NaN
        )

    # Tạo khóa đơn vị/ao
    if all(col in df.columns for col in ['Vụ nuôi', 'module_name', 'ao']):
        df['units'] = df.apply(
            lambda x: f"{str(x['Vụ nuôi']).replace(' ', '')}-{x['module_name']}-{x['ao']}", axis=1
        )
        df.drop(['Vụ nuôi', 'module_name', 'ao'], axis=1, inplace=True)

    # Sắp xếp và loại NA
    df.sort_values(['units', 'Date'], inplace=True)
    df.dropna(axis=0, inplace=True)
    return df


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
    if ':' in s:
        parts = s.split(':')
        try:
            h = int(parts[0])
            m = int(float(parts[1]))
            return (h, m)
        except Exception:
            return (np.nan, np.nan)
    return (np.nan, np.nan)


def create_morning_afternoon_same_day_dataset(df: pd.DataFrame, column: str = 'Độ kiềm') -> pd.DataFrame:
    """
    Tạo dataset dự đoán Độ kiềm buổi chiều (afternoon, >15h) trong cùng ngày từ bản ghi buổi sáng.
    Quy tắc nhóm theo giờ trong 'Time':
      - hour < 9        -> 'morning'
      - 9 <= hour <= 15 -> 'noon'
      - hour > 15       -> 'afternoon'
    Chỉ giữ các ngày có cả morning và afternoon; dùng bản ghi cuối trong mỗi nhóm.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    work = df.copy()
    work['Date'] = pd.to_datetime(work['Date'])
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
    work = work[work['time_bucket'].isin(['morning', 'afternoon'])].copy()

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

    # Loại cột tạm nếu còn sót
    for col in ['_dt', '_hour', 'time_bucket', 'Time']:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    return merged.dropna(axis=0)


def preprocessingdata(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.drop(categorical_col, axis=1).copy()
    df1 = df[(np.abs(stats.zscore(df_num)) < zscore_lim).all(axis=1)].copy()

    # Giữ lại các cột cần thiết cho huấn luyện
    selected_cols = input_col_list + output_column
    # Bổ sung các cột lag nếu có
    lag_cols = [col for col in df.columns if 'lag' in col]
    selected_cols += lag_cols

    df1 = df1[selected_cols + [c for c in categorical_usecol_all if c in df.columns]].copy()
    df1.reset_index(drop=True, inplace=True)

    # One hot encoder cho biến phân loại
    oh_cols = [c for c in categorical_usecol_all if c in df1.columns]
    if oh_cols:
        oh_enc = OneHotEncoder(sparse_output=False)
        oh_enc.fit(df1[oh_cols])
        oh_df = pd.DataFrame(oh_enc.transform(df1[oh_cols]), columns=oh_enc.get_feature_names_out())
        df1 = pd.concat([oh_df, df1], axis=1)
        df1.drop(oh_cols, inplace=True, axis=1)

    return df1


def RandomForest_random_cv_joint(
    X: pd.DataFrame,
    y: pd.DataFrame,
    y_today: pd.DataFrame,
    n_splits: int = 10,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[dict, List[float], List[float]]:
    """
    Cross-Validation 2 bước:
    - Bước 1: Dự đoán Độ kiềm hiện tại (today) từ X.
    - Bước 2: Thêm đặc trưng 'alkaline_today_pred' vào X và dự đoán target buổi chiều cùng ngày.
    """
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    metrics_result = {"RMSE": [], "MAE": [], "MAPE": [], "MPE": [], "ME": [], "R2": []}
    y_test_all: List[float] = []
    y_pred_all: List[float] = []

    fold = 0
    for train_index, test_index in splitter.split(X):
        fold += 1
        # split
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_today_train, y_today_test = y_today.iloc[train_index], y_today.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Step 1: predict today
        X_scaler_today = StandardScaler()
        y_scaler_today = StandardScaler()
        X_train_scaled_today = X_scaler_today.fit_transform(X_train)
        y_train_scaled_today = y_scaler_today.fit_transform(y_today_train.values.reshape(-1, 1))
        model_alkaline_today = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=5,
            max_features='sqrt', random_state=fold, bootstrap=False, verbose=0
        )
        model_alkaline_today.fit(X_train_scaled_today, y_train_scaled_today.ravel())

        alkaline_train_pred = y_scaler_today.inverse_transform(
            model_alkaline_today.predict(X_train_scaled_today).reshape(-1, 1)
        )
        alkaline_test_pred = y_scaler_today.inverse_transform(
            model_alkaline_today.predict(X_scaler_today.transform(X_test)).reshape(-1, 1)
        )
        X_train['alkaline_today_pred'] = alkaline_train_pred.flatten()
        X_test['alkaline_today_pred'] = alkaline_test_pred.flatten()

        # Step 2: predict afternoon target
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
        model_alkaline_afternoon = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=5,
            max_features='sqrt', random_state=fold, bootstrap=False, verbose=0
        )
        model_alkaline_afternoon.fit(X_train_scaled, y_train_scaled.ravel())

        y_pred_scaled = model_alkaline_afternoon.predict(X_scaler.transform(X_test)).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_test.values.reshape(-1, 1)

        # metrics
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        err = y_pred.flatten() - y_true.flatten()
        me = float(np.mean(err))
        denom = np.where(np.abs(y_true.flatten()) > 1e-8, y_true.flatten(), 1e-8)
        mpe = float(np.mean(err / denom) * 100.0)

        metrics_result['RMSE'].append(rmse)
        metrics_result['MAE'].append(mae)
        metrics_result['MAPE'].append(mape)
        metrics_result['MPE'].append(mpe)
        metrics_result['ME'].append(me)
        metrics_result['R2'].append(r2)

        y_test_all.extend(y_true.flatten())
        y_pred_all.extend(y_pred.flatten())

    return metrics_result, y_test_all, y_pred_all


def plot_walk_forward_metrics(metrics_result: dict, save_path: str = 'walk_forward_metrics.png'):
    if not metrics_result:
        return
    cols = 3 if len(metrics_result) > 4 else 2
    rows = int(np.ceil(len(metrics_result) / cols))
    plt.figure(figsize=(5 * cols, 3.5 * rows))
    for idx, (key, values) in enumerate(metrics_result.items(), 1):
        plt.subplot(rows, cols, idx)
        plt.plot(values, marker='o')
        plt.title(f"{key} per Walk-Forward Step")
        plt.xlabel('Step')
        plt.ylabel(key)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions_over_time(y_true: List[float], y_pred: List[float], save_path: str = 'predictions_over_time.png'):
    plt.figure(figsize=(16, 5))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title('Actual vs Predicted Values Over Time (Walk-Forward)', fontsize=14)
    plt.xlabel('Time Step')
    plt.ylabel('Target')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions_sorted_by_groundtruth(y_true: List[float], y_pred: List[float], save_path: str = 'predictions_sorted_by_groundtruth.png'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sorted_idx = np.argsort(y_true)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_sorted, label='Actual (Sorted)', color='blue', linewidth=2)
    plt.plot(y_pred_sorted, label='Predicted (Sorted by Actual)', color='orange', linestyle='', marker='2')
    plt.title('Actual vs Predicted Values Sorted by Actual', fontsize=14)
    plt.xlabel('Index (Sorted by Actual Value)')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    pd.options.display.float_format = '{:.6f}'.format

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, f"randomforest2step_{datetime.now().strftime('%y%m%d-%H%M%S')}.log")
    global logger
    logger = setup_logger(log_path)

    df = readdata()
    df = datacleaning(df)

    global input_col
    global categorical_usecol
    input_col = input_col_list
    categorical_usecol = [c for c in categorical_usecol_all if c in input_col]

    # Dùng target chiều cùng ngày, feature từ sáng
    global output_column
    output_column = ['Độ kiềm_afternoon']

    df = create_morning_afternoon_same_day_dataset(df, column='Độ kiềm')
    df = preprocessingdata(df)

    df.to_csv(os.path.join(output_folder, 'databeforetrain1.csv'), index=False)

    X = df.drop(output_column, axis=1)
    y = df[output_column]
    y_today = df['Độ kiềm'] if 'Độ kiềm' in df.columns else df.iloc[:, X.shape[1]]

    metrics_result, y_true, y_pred = RandomForest_random_cv_joint(X, y, y_today, n_splits=50)

    ts = datetime.now().strftime('%y%m%d-%H%M%S')
    plot_walk_forward_metrics(metrics_result, save_path=f"{output_folder}/metrics_{ts}.png")
    plot_predictions_over_time(y_true, y_pred, save_path=f"{output_folder}/predictions_over_time_{ts}.png")
    plot_predictions_sorted_by_groundtruth(y_true, y_pred, save_path=f"{output_folder}/predictions_sorted_{ts}.png")


if __name__ == '__main__':
    print('Running main Program')
    main()

