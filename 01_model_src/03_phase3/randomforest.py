import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
             ]


input_col_list = [
    
    ['Nhiệt độ', 'Độ màu', 'area', 'Độ mặn', 'Loại ao', 'Độ cứng', 'TDS', 'pH', 'Tuổi tôm'],
    ['Nhiệt độ', 'Độ màu', 'area', 'Độ mặn', 'Loại ao', 'Độ cứng', 'TDS', 'pH', 'Tuổi tôm', 'Độ kiềm'],
    ['Độ màu', 'area', 'Độ mặn', 'Loại ao', 'Độ cứng', 'TDS', 'pH', 'Tuổi tôm'],
    ['Độ màu', 'area', 'Độ mặn', 'Loại ao', 'Độ cứng', 'TDS', 'pH', 'Tuổi tôm', 'Độ kiềm']

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
    Create lag features for the provided column and a next-day target.
    The transformation is applied per unit after sorting by Date.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    log_and_flush(logger, f"Creating lag features for '{column}', window size = {window_size}")
    log_and_flush(logger, f"Original shape: {df.shape}")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    df_sorted = df.sort_values(by=['units', 'Date'])
    lagged_dfs = []

    for unit, group in df_sorted.groupby('units'):
        group = group.sort_values(by='Date').reset_index(drop=True)

        # Create lag features
        for i in range(1, window_size + 1):
            group[f'{column}_lag_{i}'] = group[column].shift(i)

        # Create target column shifted to the next day
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

    # Rename column 'Amoni' to 'TAN'
    df.rename({'Amoni': 'TAN'}, axis=1, inplace=True)

    return df

def datacleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Replace zero water-level readings with NaN before imputation
    df.loc[df['Mực nước'] == 0, 'Mực nước'] = np.NaN
    df['Mực nước'].fillna(df['Mực nước'].median(), inplace=True)

    # Drop the 'Time' column
    df.drop(['Time'], axis=1, inplace=True)

    # Parse the 'Date' column
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Convert the 'Tuổi tôm' column to numeric and mark invalid cells
    df['Tuổi tôm'] = df['Tuổi tôm'].apply(
        lambda x: int(float(x)) if str(x).replace('.', '', 1).isnumeric() else np.NaN
    )

    df['units'] = df.apply(
        lambda x: f"{x['Vụ nuôi'].replace(' ', '')}-{x['module_name']}-{x['ao']}",
        axis=1
    )
    df.drop(['Vụ nuôi', 'module_name', 'ao'], axis=1, inplace=True)

    # Sort data by unit and date
    df.sort_values(['units', 'Date'], inplace=True)

    # Drop rows containing NaN
    df.dropna(axis=0, inplace=True)

    return df

def preprocessingdata(df: pd.DataFrame) -> pd.DataFrame:
    print("----- Preprocessing -----")
    df_num = df.drop(categorical_col, axis=1).copy()

    df1 = df[(np.abs(stats.zscore(df_num)) < zscore_lim).all(axis=1)].copy()

    # Keep model inputs, targets, and any lag features
    selected_cols = input_col + output_column

    lag_cols = [col for col in df.columns if 'lag' in col]
    selected_cols += lag_cols

    df1 = df[selected_cols + categorical_usecol].copy()

    df1.reset_index(drop=True, inplace=True)

    print("One-hot encoder")
    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(df1[categorical_usecol])
    oh_df = pd.DataFrame(
        oh_enc.transform(df1[categorical_usecol]),
        columns=oh_enc.get_feature_names_out()
    )
    print(oh_df.columns)
    df1 = pd.concat([oh_df, df1], axis=1)
    df1.drop(categorical_usecol, axis=1, inplace=True)

    return df1


def RandomForest_random_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
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
    log_and_flush(logger, f"Input columns: {list(X.columns)}")
    log_and_flush(logger, f"CV Folds: {n_splits}, Test size: {test_size}")

    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    metrics_result = {"RMSE": [], "MAE": [], "MAPE": [], "R2": []}
    y_test_all = []
    y_pred_all = []

    fold = 0
    for train_index, test_index in splitter.split(X):
        fold += 1
        start_time = time.time()

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Standardize features and targets
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = X_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=fold,
            bootstrap=False,
            verbose=0
        )

        rf.fit(X_train_scaled, y_train_scaled.ravel())

        y_pred_scaled = rf.predict(X_scaler.transform(X_test)).reshape(-1, 1)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_test.values.reshape(-1, 1)

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)

        metrics_result["RMSE"].append(rmse)
        metrics_result["MAE"].append(mae)
        metrics_result["MAPE"].append(mape)
        metrics_result["R2"].append(r2)

        y_test_all.extend(y_true.flatten())
        y_pred_all.extend(y_pred.flatten())

        log_and_flush(logger, f"[Fold {fold}] RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.3f}, R2: {r2:.3f}")
        log_and_flush(logger, f"[Fold {fold}] Completed in {time.time() - start_time:.2f} seconds")
    # Plot sample predictions for the last fold
    plot_predictions_sorted_by_groundtruth(y_true.flatten(), y_pred.flatten(),
                    save_path=f"{output_folder}/predictions_groundtruth_sorted_{currenttime.strftime('%y%m%d-%H%M%S')}.png")
    # Summary statistics across folds
    log_and_flush(logger, "----- Summary (Mean +/- Std) -----")
    for metric in metrics_result:
        mean_val = np.mean(metrics_result[metric])
        std_val = np.std(metrics_result[metric])
        log_and_flush(logger, f"{metric}: {mean_val:.3f} +/- {std_val:.3f}")

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

def plot_predictions_sorted_by_groundtruth(y_true: List[float], 
                                           y_pred: List[float], 
                                           save_path: str = "predictions_sorted_by_groundtruth.png"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
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
    for _input_col in input_col_list:
        log_and_flush(logger,f"Input: {_input_col}")
        df = readdata()
        df = datacleaning(df)
        input_col = _input_col
        categorical_usecol = [_col for _col in categorical_usecol_all if _col in _input_col]
        print(f"{categorical_usecol=}")
        df = create_lag_features_and_target_tomorrow(df,window_size=0)
        df = preprocessingdata(df)
        
        df.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

        X = df.drop(output_column, axis=1)
        print()
        print(f"{X.columns=}")
        y = df[output_column]
        metrics_result, y_true, y_pred = RandomForest_random_cv(
        X, y,
        n_splits=50,
        test_size=0.3
        )
        log_and_flush(logger, f"Plotting..., time subfix: {currenttime.strftime('%y%m%d-%H%M%S')}")

        plot_walk_forward_metrics(metrics_result, save_path=f"{output_folder}/metrics_{currenttime.strftime('%y%m%d-%H%M%S')}.png")



if __name__ == "__main__":
    print("Running main Program")
    noname()
