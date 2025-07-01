import os
import sys
from scipy import stats
from matplotlib.colors import LogNorm, Normalize
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    return rmse, mae, r2, mape


def train_and_save_model(model_class, best_params, X_train, y_train, model_path):
    params = best_params.copy()
    if model_class == RandomForestRegressor or model_class == DecisionTreeRegressor:
        if 'max_depth' in params:
            params['max_depth'] = params['max_depth'] + 1
        if 'max_features' in params:
            params['max_features'] = min(params['max_features'] + 1, X_train.shape[1])
            params['max_features'] = max(params['max_features'], 1)
    elif model_class == lgb.LGBMRegressor:
        if 'max_depth' in params:
            params['max_depth'] = params['max_depth'] + 3
        if 'num_leaves' in params:
            params['num_leaves'] = params['num_leaves'] + 5
        if 'n_estimators' in params:
            params['n_estimators'] = params['n_estimators'] + 5
    elif model_class == XGBRegressor:
        if 'max_depth' in params:
            params['max_depth'] = params['max_depth'] + 3
        if 'n_estimators' in params:
            params['n_estimators'] = params['n_estimators'] + 5
    model = model_class(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model


def model_opt(model_class, hyper_space, groups, X_train, y_train, max_evals=100):
    def objective(hyperparams):
        if model_class == RandomForestRegressor or model_class == DecisionTreeRegressor:
            if 'max_features' in hyperparams:
                hyperparams['max_features'] = min(hyperparams['max_features'] + 1, X_train.shape[1])
                hyperparams['max_features'] = max(hyperparams['max_features'], 1)
        model = model_class(**hyperparams)
        gkf = GroupKFold(n_splits=5)
        val_rmse, val_mae, val_r2 = [], [], []

        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
            val_x, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]
            model.fit(train_x, train_y)
            preds = model.predict(val_x)
            val_rmse.append(np.sqrt(mean_squared_error(val_y, preds)))
            val_mae.append(mean_absolute_error(val_y, preds))
            val_r2.append(r2_score(val_y, preds))

        avg_rmse = np.mean(val_rmse)
        return {
            'loss': avg_rmse,
            'status': STATUS_OK,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2
        }

    trials = Trials()
    best = fmin(fn=objective, space=hyper_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(f"The best hyperparameters for {model_class.__name__}: {best}")
    inner_loop_performance = {
        'rmse': trials.best_trial['result']['val_rmse'],
        'mae': trials.best_trial['result']['val_mae'],
        'r2': trials.best_trial['result']['val_r2']
    }
    return best, trials, inner_loop_performance

if __name__ == "__main__":
    base_path = Path(r"\external_val\final")
    df = pd.read_csv(base_path / "scaler_features_model_train.csv", encoding='unicode_escape')
    missing_cols = df.columns[df.isna().any()].tolist()
    print("missing_cols:", missing_cols)

    X = df.iloc[:, 1:-1]
    y = df['Output_diss_fraction']
    groups = df['formulation_id']

    #scaler = StandardScaler()
    #X1 = scaler.fit_transform(X)
    #joblib.dump(scaler, base_path / "scaler.pkl")

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
            'max_features': hp.choice('max_features', list(range(1, 30))),
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

    best_models_info = {
        model_name: {
            'model': None,
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': -float('inf'),
            'fold_idx': -1,
            'params': None
        } for model_name in models.keys()
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
        lgb_model.fit(X, y)

    outer_cv = GroupKFold(n_splits=5)
    all_test_preds = {model_name: [] for model_name in models.keys()}
    all_test_y = {model_name: [] for model_name in models.keys()}
    outer_loop_performance = {model_name: {'rmse': [], 'mae': [], 'r2': []} for model_name in models.keys()}
    inner_loop_performance_all = {model_name: {'rmse': [], 'mae': [], 'r2': []} for model_name in models.keys()}

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        print(f"\nOuter Fold {fold_idx + 1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        for model_name, (model_class, hyper_space) in models.items():
            print(f"\nOptimizing {model_name} in Fold {fold_idx + 1}...")
            best_params, trials, inner_loop_performance = model_opt(model_class, hyper_space, groups_train, X_train, y_train)
            #model = train_and_save_model(model_class, best_params, X_train, y_train,base_path / f"best_{model_name}_fold{fold_idx + 1}.pkl")

            params = best_params.copy()
            if model_class == RandomForestRegressor or model_class == DecisionTreeRegressor:
                if 'max_depth' in params:
                    params['max_depth'] = params['max_depth'] + 1
                if 'max_features' in params:
                    params['max_features'] = min(params['max_features'] + 1, X_train.shape[1])
                    params['max_features'] = max(params['max_features'], 1)
            elif model_class == lgb.LGBMRegressor:
                if 'max_depth' in params:
                    params['max_depth'] = params['max_depth'] + 3
                if 'num_leaves' in params:
                    params['num_leaves'] = params['num_leaves'] + 5
                if 'n_estimators' in params:
                    params['n_estimators'] = params['n_estimators'] + 5
            elif model_class == XGBRegressor:
                if 'max_depth' in params:
                    params['max_depth'] = params['max_depth'] + 3
                if 'n_estimators' in params:
                    params['n_estimators'] = params['n_estimators'] + 5
            model = model_class(**params)
            model.fit(X_train, y_train)
            inner_loop_performance_all[model_name]['rmse'].extend(inner_loop_performance['rmse'])
            inner_loop_performance_all[model_name]['mae'].extend(inner_loop_performance['mae'])
            inner_loop_performance_all[model_name]['r2'].extend(inner_loop_performance['r2'])

            ypreds_test = model.predict(X_test)
            all_test_preds[model_name].extend(ypreds_test)
            all_test_y[model_name].extend(y_test)

            rmse_test, mae_test, r2_test, _ = eval_metrics(y_test, ypreds_test)
            outer_loop_performance[model_name]['rmse'].append(rmse_test)
            outer_loop_performance[model_name]['mae'].append(mae_test)
            outer_loop_performance[model_name]['r2'].append(r2_test)

            if rmse_test < best_models_info[model_name]['rmse']:
                best_models_info[model_name]['model'] = model
                best_models_info[model_name]['rmse'] = rmse_test
                best_models_info[model_name]['mae'] = mae_test
                best_models_info[model_name]['r2'] = r2_test
                best_models_info[model_name]['fold_idx'] = fold_idx + 1
                best_models_info[model_name]['params'] = best_params

    print("\n=== Global Best Models ===")
    for model_name, info in best_models_info.items():
        if info['model'] is not None:
            best_model_path = base_path / f"best_{model_name}_fold{info['fold_idx']}_global.pkl"
            joblib.dump(info['model'], best_model_path)
            print(f"\nBest {model_name.upper()} model saved to {best_model_path}")
            print(f"Best performance (Fold {info['fold_idx']}): RMSE = {info['rmse']:.3f}, MAE = {info['mae']:.3f}, R² = {info['r2']:.3f}")
            print(f"Best hyperparameters: {info['params']}")

    print("\n=== Performance Statistics ===")
    for model_name in models.keys():
        print(f"\nResults for {model_name.upper()}:")

        print("Inner Loop Performance (across all 25 inner folds):")
        for metric in ['rmse', 'mae', 'r2']:
            mean_val = np.mean(inner_loop_performance_all[model_name][metric])
            std_val = np.std(inner_loop_performance_all[model_name][metric])
            print(f"  {metric.upper()}: {mean_val:.3f}±{std_val:.3f}")

        print("Outer Loop Performance (across 5 outer folds):")
        for metric in ['rmse', 'mae', 'r2']:
            mean_val = np.mean(outer_loop_performance[model_name][metric])
            std_val = np.std(outer_loop_performance[model_name][metric])
            print(f"  {metric.upper()}: {mean_val:.3f}±{std_val:.3f}")

    for model_name in models.keys():
        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(4, 4)

        # Clip the data to the range [0, 1]
        clipped_test_y = np.clip(all_test_y[model_name], 0, 1)  # Clip experimental values
        clipped_test_preds = np.clip(all_test_preds[model_name], 0, 1)  # Clip predicted values

        # Main scatter plot (hexbin)
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        slope_test, intercept_test, r_value_test, _, _ = stats.linregress(clipped_test_y, clipped_test_preds)
        best_line_x = np.linspace(0, 1, 100)  # Use [0, 1] for the best-fit line
        best_line_y = best_line_x
        hb = ax_main.hexbin(clipped_test_y, clipped_test_preds, gridsize=50,
                            cmap='Blues', mincnt=1, norm=LogNorm())
        ax_main.plot(best_line_x, best_line_y, color='k', linewidth=1, alpha=0.8, linestyle='--', label='y=x')

        # Calculate metrics and add text annotations
        rmse_test = np.mean(outer_loop_performance[model_name]['rmse'])
        r2_test = np.mean(outer_loop_performance[model_name]['r2'])
        # Use fixed positions for text to avoid issues with clipped data
        ax_main.text(0.02, 0.95, f'N = {len(clipped_test_y):,}', fontsize=10)
        ax_main.text(0.02, 0.88, f'y = {slope_test:.2f}x + {intercept_test:.2f}', fontsize=10)
        ax_main.text(0.02, 0.81, f'$R^2$ = {r2_test:.3f}', fontsize=10)
        ax_main.text(0.02, 0.74, f'Pearson = {r_value_test:.3f}', fontsize=10)
        ax_main.text(0.02, 0.67, f'RMSE = {rmse_test:.3f}', fontsize=10)
        ax_main.set_xlabel('Experimental Value', fontsize=12)
        ax_main.set_ylabel('Predicted Value', fontsize=12)

        # Set axis limits to [0, 1]
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)

        # X-axis histogram (Experimental Value)
        ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        n, bins, patches = ax_histx.hist(clipped_test_y, bins=50, range=(0, 1))  # Ensure histogram respects [0, 1]
        cm = plt.get_cmap('Blues')
        norm = Normalize(vmin=min(n), vmax=max(n))
        for c, p in zip(n, patches):
            plt.setp(p, 'facecolor', cm(norm(c)))
        ax_histx.set_xlim(0, 1)  # Explicitly set limits to match ax_main
        ax_histx.set_yticks([])
        for spine in ax_histx.spines.values():
            spine.set_visible(False)

        # Y-axis histogram (Predicted Value)
        ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
        n, bins, patches = ax_histy.hist(clipped_test_preds, bins=50, range=(0, 1),
                                         orientation='horizontal')  # Ensure histogram respects [0, 1]
        cm = plt.get_cmap('Blues')
        norm = Normalize(vmin=min(n), vmax=max(n))
        for c, p in zip(n, patches):
            plt.setp(p, 'facecolor', cm(norm(c)))
        ax_histy.set_ylim(0, 1)  # Explicitly set limits to match ax_main
        ax_histy.set_xticks([])
        for spine in ax_histy.spines.values():
            spine.set_visible(False)

        # Add colorbar
        fig.colorbar(hb, ax=ax_main, label='Count', fraction=0.046, pad=0.04)

        # Add title and adjust layout
        plt.suptitle(f'{model_name.upper()} Test Set Predictions', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(base_path / f'{model_name}_Test_Set_scatter_hist.png', dpi=600)
        plt.close()

    print("All model evaluations and plots completed!")

