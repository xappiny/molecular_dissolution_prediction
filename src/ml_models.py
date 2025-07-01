import os
import sys
from scipy import stats
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, GroupKFold,RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import warnings
import lightgbm

warnings.filterwarnings('ignore')
import sys
import os


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    return rmse, mae, r2, mape


def model_opt(model_class, hyper_space, groups, X_train, y_train, max_evals=100):
    def objective(hyperparams):
        model = model_class(**hyperparams)
        gkf = RepeatedKFold(n_splits=5,n_repeats=2,random_state=42)
        val_rmse, val_mae, val_r2 = [], [], []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            train_x, train_y = X_train[train_idx], y_train.iloc[train_idx]
            val_x, val_y = X_train[val_idx], y_train.iloc[val_idx]
            model.fit(train_x, train_y)
            preds = model.predict(val_x)

            val_rmse.append(np.sqrt(mean_squared_error(val_y, preds)))
            val_mae.append(mean_absolute_error(val_y, preds))
            val_r2.append(r2_score(val_y, preds))

        avg_rmse = np.mean(val_rmse)
        std_rmse = np.std(val_rmse)
        avg_mae = np.mean(val_mae)
        std_mae = np.std(val_mae)
        avg_r2 = np.mean(val_r2)
        std_r2 = np.std(val_r2)

        return {'loss': avg_rmse, 'status': STATUS_OK, 'avg_rmse': avg_rmse, 'avg_mae': avg_mae, 'avg_r2': avg_r2,
                'std_rmse': std_rmse, 'std_mae': std_mae, 'std_r2': std_r2}

    trials = Trials()
    best = fmin(fn=objective, space=hyper_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    print(f"The best hyperparameters are: {best}")
    best_loss = min([x['result']['loss'] for x in trials.trials])
    best_rmse_mean = trials.best_trial['result']['avg_rmse']
    best_rmse_std = trials.best_trial['result']['std_rmse']
    best_mae_mean = trials.best_trial['result']['avg_mae']
    best_mae_std = trials.best_trial['result']['std_mae']
    best_r2_mean = trials.best_trial['result']['avg_r2']
    best_r2_std = trials.best_trial['result']['std_r2']

    print(f"Best loss: {best_loss:.3f}")
    print(f"Best Avg RMSE: {best_rmse_mean:.3f} ± {best_rmse_std:.3f}")
    print(f"Best Avg MAE: {best_mae_mean:.3f} ± {best_mae_std:.3f}")
    print(f"Best Avg R2: {best_r2_mean:.3f} ± {best_r2_std:.3f}")

    return best, trials


def train_and_save_model(model_class, best_params, X_train, y_train, model_path):
    model = model_class(**best_params)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model


def show_predict_performance(X_train, y_train, X_test, y_test, model, plot=True, train_title='Training Set', test_title='Test Set'):
    ypreds_train = model.predict(X_train)
    ypreds_test = model.predict(X_test)

    print(f"{train_title} result")
    rmse_train, mae_train, r2_train, _ = eval_metrics(y_train, ypreds_train)
    print(f"  RMSE: {rmse_train:.3f}")
    print(f"  MAE: {mae_train:.3f}")
    print(f"  R^2: {r2_train:.3f}")

    print(f"{test_title} result")
    rmse_test, mae_test, r2_test, _ = eval_metrics(y_test, ypreds_test)
    print(f"  RMSE: {rmse_test:.3f}")
    print(f"  MAE: {mae_test:.3f}")
    print(f"  R^2: {r2_test:.3f}")

    if plot:
        # 绘制训练集的散点密度图
        fig_train = plt.figure(figsize=(8, 6))
        slope_train, intercept_train, _, _, _ = stats.linregress(y_train, ypreds_train)
        best_line_x_train = np.linspace(min(y_train), max(y_train), 100)
        best_line_y_train = best_line_x_train
        plt.hexbin(y_train, ypreds_train, gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
        cb = plt.colorbar(label='Counts')
        plt.plot(best_line_x_train, best_line_y_train, color='k', linewidth=1, alpha=0.8, linestyle='-', label='y=x')
        plt.text(min(y_train), max(ypreds_train) * 0.95, f'y={round(slope_train, 3)}x + {round(intercept_train, 3)}', fontsize=10)
        plt.text(min(y_train), max(ypreds_train) * 0.88, f'$R^2$={round(r2_train, 2)}', fontsize=10)
        plt.text(min(y_train), max(ypreds_train) * 0.81, f'RMSE={round(rmse_train, 2)}', fontsize=10)
        plt.text(min(y_train), max(ypreds_train) * 0.74, f'MAE={round(mae_train, 2)}', fontsize=10)
        plt.xlabel('True Values', fontsize=14)
        plt.ylabel('Predict Values', fontsize=14)
        plt.title(train_title, fontsize=16)
        plt.legend(loc="lower right", markerscale=1.5, frameon=False, handletextpad=-.1)
        plt.tight_layout()
        plt.savefig(f'{train_title}_scatter.png', dpi=600)
        plt.show()

        # 绘制测试集的散点密度图
        fig_test = plt.figure(figsize=(8, 6))
        slope_test, intercept_test, _, _, _ = stats.linregress(y_test, ypreds_test)
        best_line_x_test = np.linspace(min(y_test), max(y_test), 100)
        best_line_y_test = best_line_x_test
        plt.hexbin(y_test, ypreds_test, gridsize=50, cmap='Blues', mincnt=1, norm=LogNorm())
        cb_test = plt.colorbar(label='Counts')
        plt.plot(best_line_x_test, best_line_y_test, color='k', linewidth=1, alpha=0.8, linestyle='-', label='y=x')
        plt.text(min(y_test), max(ypreds_test) * 0.95, f'y={round(slope_test, 3)}x + {round(intercept_test, 3)}', fontsize=10)
        plt.text(min(y_test), max(ypreds_test) * 0.88, f'$R^2$={round(r2_test, 2)}', fontsize=10)
        plt.text(min(y_test), max(ypreds_test) * 0.81, f'RMSE={round(rmse_test, 2)}', fontsize=10)
        plt.text(min(y_test), max(ypreds_test) * 0.74, f'MAE={round(mae_test, 2)}', fontsize=10)
        plt.xlabel('True Values', fontsize=14)
        plt.ylabel('Predict Values', fontsize=14)
        plt.title(test_title, fontsize=16)
        plt.legend(loc="lower right", markerscale=1.5, frameon=False, handletextpad=-.1)
        plt.tight_layout()
        plt.savefig(f'{test_title}_scatter.png', dpi=600)
        plt.show()

    scores = {
        'R2_test': r2_test,
        'RMSE_test': rmse_test,
        'MAE_test': mae_test
    }

    return scores


if __name__ == "__main__":
    # 读取数据
    base_path = Path(r"C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution")
    df = pd.read_csv(base_path / "dissolution_features.csv", encoding='unicode_escape')

    missing_cols = df.columns[df.isna().any()].tolist()
    print("包含缺失值的列:", missing_cols)

    # 使用GroupShuffleSplit进行组随机切分
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
    for train_idx, val_idx in gss.split(df, groups=df['formulation_id']):
        train_prescriptions, val_prescriptions = df.iloc[train_idx]['formulation_id'], df.iloc[val_idx]['formulation_id']

    # 根据划分得到的处方编号获取对应的数据
    train_set = df[df['formulation_id'].isin(train_prescriptions)]
    test_set = df[df['formulation_id'].isin(val_prescriptions)]

    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    train_set.to_csv(base_path / "train_set.csv", index=False)
    test_set.to_csv(base_path / "test_set.csv", index=False)

    X_train1 = train_set.iloc[:, 2:]
    y_train = train_set['diss_fraction']
    groups = train_set['formulation_id']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train1)
    joblib.dump(scaler, base_path / "scaler.pkl")

    # 定义模型和超参数空间
    models = {
        'lgb': (lgb.LGBMRegressor, {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
            'max_depth': hp.choice('max_depth', range(3, 15)),
            'num_leaves': hp.choice('num_leaves', range(5, 256)),
            'subsample': hp.uniform('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
            'n_estimators': hp.choice('n_estimators', range(5, 2000)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 100),
            "reg_lambda": hp.uniform("reg_lambda", 0, 100)
        }),
        'rf': (RandomForestRegressor, {
            'max_depth': hp.choice('max_depth', range(1, 100)),
            'max_features': hp.choice('max_features', range(1, 30)),
            'n_estimators': hp.choice('n_estimators', range(100, 1000))
        }),
        'xgb': (XGBRegressor, {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.2),
            'max_depth': hp.choice('max_depth', range(3, 20)),
            'subsample': hp.uniform('subsample', 0.6, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
            'n_estimators': hp.choice('n_estimators', range(5, 1000)),
            "reg_alpha": hp.uniform("reg_alpha", 0, 100),
            "reg_lambda": hp.uniform("reg_lambda", 0, 100),
            'min_child_weight': hp.randint('min_child_weight', 6),
            'gamma': hp.uniform('gamma', 0, 1)
        }),
        'mlp': (MLPRegressor, {
            'alpha': hp.loguniform('alpha', -6, 0),
        }),
        'dt': (DecisionTreeRegressor, {
            'max_depth': hp.choice('max_depth', range(1, 100)),
            'max_features': hp.choice('max_features', range(1, 30))
        })
    }


    class SuppressOutput:
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr


    with SuppressOutput():
        lgb_model = lgb.LGBMRegressor(verbose=-1)
        lgb_model.fit(X_train, y_train)

    # 训练和保存模型
    best_models = {}
    for model_name, (model_class, hyper_space) in models.items():
        best_params, _ = model_opt(model_class, hyper_space, groups, X_train, y_train)
        model = train_and_save_model(model_class, best_params, X_train, y_train, base_path / f"best_{model_name}.pkl")
        best_models[model_name] = model

    # 加载测试集
    test_data = pd.read_csv(base_path / "test_set.csv", encoding='unicode_escape')
    X_test  = test_set.iloc[:, 2:]
    y_test = test_set["diss_fraction"]
    X_test = scaler.transform(X_test)

    # 评估模型性能
    for model_name, model in best_models.items():
        show_predict_performance(X_train, y_train, X_test, y_test, model, plot=True, train_title=f'{model_name.upper()} Training Set', test_title=f'{model_name.upper()} Test Set')

    # 使用各个模型进行预测
    predictions = {model_name: model.predict(X_test) for model_name, model in best_models.items()}
    output_df = pd.DataFrame(predictions)
    output_df.to_csv(base_path / "model_predictions.csv", index=False)