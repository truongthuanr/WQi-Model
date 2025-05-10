from helper import consolelog
# from config import columns, input_col, categorical_col, output_column, categorical_usecol
# from config import output_folder
# from config import zscore_lim



import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from datetime import datetime

import pandas as pd
import numpy as np
import shap
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
import logging



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


input_col = [
     # thông số đặc tính
    'Season', 
    'Loại ao', 
    'Công nghệ nuôi', 
    'Giống tôm',  
    'Mực nước',
    'Tuổi tôm',
     'area', 
    'Nhiệt độ', 'pH', 'DO', # thông số bắt buộc
    # 'Ngày thả',
    'Độ mặn', 
    'TDS', 
    'Độ đục',
    'Độ trong',
    #phase 2 
    'Độ cứng',
    'Độ màu',
    # 'Silica',
    ]

output_folder = "output"



categorical_col = ['Date',
                   'Season', 
                   'Loại ao', 
                   'Công nghệ nuôi', 
                   'Giống tôm',
                   'units']

categorical_usecol = [
    'Season', 
    'Loại ao', 
    'Công nghệ nuôi', 
    'Giống tôm'
    ]

# output_column = ['TAN', 'Nitrat', 'Nitrit', 'Silica', 'Canxi', 'Kali', 'Magie', 'Độ kiềm', 'Độ cứng']
output_column = ['Độ kiềm']
zscore_lim =  3



def loginit():
    global logger 
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{output_folder}/feature_importance.log', 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        encoding='utf-8')

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
    consolelog("Read data!")
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

    df1 = df1[input_col+output_column].copy()
    df1.reset_index(drop=True,inplace=True)

    df1.to_csv(os.path.join(output_folder,"databeforetrain1.csv"))

    consolelog("Plot data!")
    plt.figure(figsize=(10,5))
    sns.boxenplot(df1)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
    # plt.show()
    plt.savefig(os.path.join(output_folder,"boxplot2.png"))


    consolelog("One hot encorder")
    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(df1[categorical_usecol])
    oh_df = pd.DataFrame(oh_enc.transform(df1[categorical_usecol]),
                     columns=oh_enc.get_feature_names_out()
                    )
    print(oh_df.columns)
    df1 = pd.concat([oh_df,df1],axis=1)
    df1.drop(categorical_usecol,inplace=True,axis=1)

    return df1


def RandomForest_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.33, random_state=1)
    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train_tf = X_sc.transform(X_train)

    y_sc = StandardScaler()
    y_sc.fit(y_train)
    y_train_tf = y_sc.transform(y_train)

    consolelog("Create RandomForeest")
    logger.info("Create RandomForeest")

    
    rf = RandomForestRegressor(n_estimators=800,
                               max_depth=50,
                               min_samples_split=2,
                               min_samples_leaf=2,
                               max_features='sqrt', 
                               random_state=0,
                               bootstrap=False,
                               verbose=1,)
    
    

    consolelog("Training...")
    rf.fit(X_train_tf,np.reshape(y_train_tf,(-1)))


    imp_feats = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    print(f"------------------")

    print(f"Feature Importance: {imp_feats}")
    logger.info(f"Feature Importance: {imp_feats=}")
    print(f"{std=}")
    logger.info(f"{std=}")

    # logger.info("Permutation Importance Random Forest")
    # r = permutation_importance(estimator=rf, 
    #                            X=X_train_tf, 
    #                            y=np.reshape(y_train_tf,(-1)),
    #                             n_repeats=200,
    #                             random_state=42)

    # logger.info(f"{X.columns=}")
    # logger.info(f"{r.importances_mean=}")
    # logger.info(f"{r.importances_std=}")
    print("----- End Random Forest Regressor --------")

    
    # Lấy 100 mẫu đại diện từ tập huấn luyện
    X_sample = X_train_tf[:100]

    # Tạo SHAP Explainer cho mô hình cây
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)

    # Plot tổng quan importance (global)
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns)
    # Lưu thành file PNG
    plt.savefig("output/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


    # rf.score(X_sc.transform(X_test),y_sc.transform(y_test))
    # y_pred = rf.predict(X_sc.transform(X_test))
    # y_pred = np.reshape(y_pred,(-1,1))
    # print(f"{type(y_pred)=}")
    # print(f"{y_pred.shape=}")
    # print(f"{y_test.to_numpy().shape=}")
    # print(f"{type(y_test.to_numpy())}")
    # plot_result(y_test=y_test.to_numpy(),
    #             y_pred=y_sc.inverse_transform(y_pred), 
    #             output_column=output_column,
    #             modelname="RandomForest")
    

    # Random searchCV
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


def GradientBoost_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.33, random_state=1)
    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train_tf = X_sc.transform(X_train)

    y_sc = StandardScaler()
    y_sc.fit(y_train)
    y_train_tf = y_sc.transform(y_train)

    consolelog("Create Gradient Boosting Regressor")
    logger.info("Create Gradient Boosting Regressor")
    
    gbr = GradientBoostingRegressor(n_estimators=800,
                               max_depth=50,
                               min_samples_split=2,
                               min_samples_leaf=2,
                               max_features='sqrt', 
                               loss='squared_error',
                               random_state=0,
                               verbose=1,)

    
    

    consolelog("Training...")
    gbr.fit(X_train_tf,np.reshape(y_train_tf,(-1)))
    r = permutation_importance(estimator=gbr, 
                               X=X_train_tf, 
                               y=np.reshape(y_train_tf,(-1)),
                                n_repeats=200,
                                random_state=42)
    print(f"{X.columns=}")
    print(f"{r.importances_mean}")
    print(f"{r.importances_std}")
    logger.info(f"{X.columns=}")
    logger.info(f"{r.importances_mean=}")
    logger.info(f"{r.importances_std=}")
    print("----- End Gradient Boosting Regressor --------")


def SVRModel(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.33, random_state=1)
    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train_tf = X_sc.transform(X_train)

    y_sc = StandardScaler()
    y_sc.fit(y_train)
    y_train_tf = y_sc.transform(y_train)

    consolelog("Create SVR Regressor")
    logger.info("Create SVR Regressor")


    SVRreg = svm.SVR(kernel='rbf',epsilon=0.5,C=10)
    # print(f"{X_train.shape=} {y_train.shape=}")
    SVRreg.fit(X_train_tf,np.reshape(y_train_tf,(-1)))

    y_pred = SVRreg.predict(X_sc.transform(X_test))
    y_pred = np.reshape(y_pred,(-1,1))


    plot_result(y_test=y_test.to_numpy(),
            y_pred=y_sc.inverse_transform(y_pred), 
            output_column=output_column,
            modelname="SVRModel")
    
    r = permutation_importance(estimator=SVRreg, 
                            #    X=X_sc.transform(X_test), 
                               X=X_train_tf, 
                            #    y=y_sc.transform(y_test),
                               y=np.reshape(y_train_tf,(-1)),

                                n_repeats=200,
                                random_state=42)
    # for i in r.importances_mean.argsort()[::-1]:
    #     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    #         print(f"{X.columns[i]:<30}"
    #           f"{r.importances_mean[i]:.3f}"
    #           f" +/- {r.importances_std[i]:.3f}")
    print(f"{X.columns=}")
    print(f"{r.importances_mean}")
    print(f"{r.importances_std}")
    logger.info(f"{X.columns=}")
    logger.info(f"{r.importances_mean=}")
    logger.info(f"{r.importances_std=}")
    
    # Random searchCV
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
    print("----- End Support Vector Regressor --------")



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
    shape = len(X.columns)
    model1.add(Input(shape=(shape,)))

    # Layer #
    model1.add(Dense(120, kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    # # Layer #
    model1.add(Dense(80,kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    # # Layer #
    model1.add(Dense(40,kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    # # Layer #
    # model1.add(Dense(16,kernel_initializer='he_uniform', activation='relu'))
    # model1.add(Dropout(0.1))
    # # Layer #
    # model1.add(Dense(16,kernel_initializer='he_uniform', activation='relu'))
    # model1.add(Dropout(0.1))
    # # Layer #
    # model1.add(Dense(16,kernel_initializer='he_uniform', activation='relu'))
    # model1.add(Dropout(0.1))
    # Layer #
    # model1.add(Dense(8, kernel_initializer='he_uniform', activation='relu'))
    # model1.add(Dropout(0.1))

    model1.add(Dense(1))
    model1.compile(loss='mae', 
                   optimizer='nadam',
                   metrics=['accuracy','r2_score']
                   )
    model1.summary()

    history = model1.fit(X_train_tf,y_train_tf,
                epochs=300,
                batch_size=32,
                verbose=False,
                validation_split=0.2)
    print(f"{model1.get_weights()[0]=}")
    print(f"{model1.get_weights()[0].shape=}")


    plot_history(history.history)

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


def wrapAnn():
    model1 = Sequential()
    model1.add(Input(shape=(21,)))
    model1.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(16,kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(8, kernel_initializer='he_uniform', activation='relu'))
    model1.add(Dropout(0.1))
    model1.add(Dense(1))
    model1.compile(loss='mae', 
                   optimizer='nadam',
                   metrics=['accuracy','r2_score']
                   )
    return model1

def permutationAnn(X,y):
    logger.info("ANN model")
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

    model = KerasRegressor(build_fn=wrapAnn,
                           epochs=100,
                           batch_size=64,
                           verbose=True
                               )
    model.fit(X_train_tf,y_train_tf)
    r = permutation_importance(model,
                                         X=X_train_tf,
                                         y=y_train_tf,
                                         random_state=42)
    print(f"{X.columns=}")
    print(f"{r.importances_mean}")
    print(f"{r.importances_std}")
    logger.info(f"{X.columns=}")
    logger.info(f"{r.importances_mean=}")
    logger.info(f"{r.importances_std=}")


def getsvrgrid():
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [0.1, 1.0, 10.0, 100.0], 
    epsilon = [0.01, 0.1, 0.2, 0.5]



    random_grid = {'kernel':kernel,
                   "C":[0.1, 1.0, 10.0, 100.0],
                   "epsilon":epsilon}
    
    return random_grid

def poly(X: pd.DataFrame, y: pd.DataFrame):
    print("======= Running poly ========") 
   
    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.33, random_state=1)
    X_sc = StandardScaler()
    X_sc.fit(X_train)
    X_train_tf = X_sc.transform(X_train)
    X_train_tf2 = X_train_tf**2
    X_train_tf_ = np.hstack((X_train_tf,X_train_tf2))
    print(f"X_train_tf shape: {np.shape(X_train_tf)}")
    print(f"X_train_tf2 shape: {np.shape(X_train_tf2)}")
    print(f"X_train_tf_ shape: {np.shape(X_train_tf_)}")


    y_sc = StandardScaler()
    y_sc.fit(y_train)
    y_train_tf = y_sc.transform(y_train)
    
    lr = LinearRegression()

    consolelog("Training...")
    lr.fit(X_train_tf_,np.reshape(y_train_tf,(-1)))

    X_test_tf = X_sc.transform(X_test)
    X_test_tf2 = X_test_tf**2
    X_test_tf_ = np.hstack((X_test_tf,X_test_tf2))


    lr.score(X_test_tf_,y_sc.transform(y_test))
    y_pred = lr.predict(X_test_tf_)
    y_pred = np.reshape(y_pred,(-1,1))

    plot_result(y_test=y_test.to_numpy(),
                y_pred=y_sc.inverse_transform(y_pred), 
                output_column=output_column,
                modelname="Linear Regression")


def noname():
    # np.set_printoptions(suppress=True)
    loginit()
    logger.info(" Start Main Program ")
    pd.options.display.float_format = '{:.6f}'.format
    df = readdata()
    # print(df.head())

    df = datacleaning(df)
    # print(df.head())

    df = preprocessingdata(df)
    # print(df.columns)
    X = df.drop(output_column, axis=1)
    print()
    print(f"{X.columns=}")
    y = df[output_column]
    # get_random_grid()

    RandomForest_model(X,y)
    # GradientBoost_model(X,y)
    # SVRModel(X,y)
    # ANNModel(X,y)
    # permutationAnn(X,y)
    # poly(X,y)



if __name__ == "__main__":
    consolelog("Running main Program")
    
    noname()
