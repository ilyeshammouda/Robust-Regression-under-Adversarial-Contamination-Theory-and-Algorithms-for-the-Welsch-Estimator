import numpy as np
from scipy.linalg import inv, det
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.robust.norms import Hampel
from sklearn.model_selection import KFold
from statsmodels.robust.robust_linear_model import RLM
from sklearn.metrics import  mean_absolute_error
from algorithms.Huber import HuberAlgo
from algorithms.Welsch_adapative_sigma import WelschAlgo as WeslchAlgo
from algorithms.Welsch_adapative_sigma import  LeastAbsoluteDeviationAlgo



def kfold_cv_tukey(X_eval, y_eval, c, n_splits=5, seed=42, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    y_oof_pred = np.empty_like(y_eval, dtype=float)

    fold_medians = []
    fold_iqrs = []
    fold_maes = []

    for train_idx, test_idx in kf.split(X_eval):
        X_tr, y_tr = X_eval[train_idx], y_eval[train_idx]
        X_te, y_te = X_eval[test_idx], y_eval[test_idx]

        tukey = sm.robust.norms.TukeyBiweight(c=c)
        model = sm.RLM(y_tr, X_tr, M=tukey)
        result = model.fit()

        y_pred_fold = X_te @ result.params
        y_oof_pred[test_idx] = y_pred_fold

        absolute_errors = np.abs(y_te - y_pred_fold)
        residuals = y_te - y_pred_fold
        iqr = np.quantile(residuals, 0.75) - np.quantile(residuals, 0.25)

        fold_medians.append(np.median(absolute_errors))
        fold_iqrs.append(iqr)
        fold_maes.append(mean_absolute_error(y_te, y_pred_fold))

    residuals = y_eval - y_oof_pred

    metrics = {
        "Tukey Biweight Metrics": {
            "Median of MedAE": float(np.median(fold_medians)),
            "Median of MeanAE": float(np.median(fold_maes)),
            "Median of IQR": float(np.median(fold_iqrs)),
        }
    }

    return y_oof_pred, residuals, metrics



def kfold_cv_huber(X_eval, y_eval, best_gamma, n_splits=5, seed=42, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    y_oof_pred = np.empty_like(y_eval, dtype=float)

    fold_medians = []
    fold_iqrs = []
    fold_maes = []

    for train_idx, test_idx in kf.split(X_eval):
        X_tr, y_tr = X_eval[train_idx], y_eval[train_idx]
        X_te, y_te = X_eval[test_idx], y_eval[test_idx]

        model = HuberAlgo(X_tr, y_tr)
        beta_hat, _ = model.optimizer_approach(
            method='L-BFGS-B',
            gamma=best_gamma,
            max_iter=100,
            initial_guess=np.zeros(X_tr.shape[1]),
        )

        y_pred_fold = X_te @ beta_hat
        y_oof_pred[test_idx] = y_pred_fold

        residuals_fold = y_te - y_pred_fold
        iqr = np.quantile(residuals_fold, 0.75) - np.quantile(residuals_fold, 0.25)

        fold_medians.append(np.median(np.abs(residuals_fold)))
        fold_iqrs.append(iqr)
        fold_maes.append(mean_absolute_error(y_te, y_pred_fold))

    residuals = y_eval - y_oof_pred

    metrics = {
        "Huber Metrics": {
            "Median of MedAE": float(np.median(fold_medians)),
            "Median of MeanAE": float(np.median(fold_maes)),
            "Median of IQR": float(np.median(fold_iqrs)),
        }
    }

    return y_oof_pred, residuals, metrics



def kfold_cv_hampel(X_eval, y_eval, best_params, n_splits=5, seed=42, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    y_oof_pred = np.empty_like(y_eval, dtype=float)

    fold_medians = []
    fold_iqrs = []
    fold_maes = []

    for train_idx, test_idx in kf.split(X_eval):
        X_tr, y_tr = X_eval[train_idx], y_eval[train_idx]
        X_te, y_te = X_eval[test_idx], y_eval[test_idx]

        model = RLM(y_tr, X_tr, M=Hampel(**best_params))
        result = model.fit()

        y_pred_fold = X_te @ result.params
        y_oof_pred[test_idx] = y_pred_fold

        residuals_fold = y_te - y_pred_fold
        iqr = np.quantile(residuals_fold, 0.75) - np.quantile(residuals_fold, 0.25)

        fold_medians.append(np.median(np.abs(residuals_fold)))
        fold_iqrs.append(iqr)
        fold_maes.append(mean_absolute_error(y_te, y_pred_fold))

    residuals = y_eval - y_oof_pred

    metrics = {
        "Hampel Metrics": {
            "Median of MedAE": float(np.median(fold_medians)),
            "Median of MeanAE": float(np.median(fold_maes)),
            "Median of IQR": float(np.median(fold_iqrs)),
        }
    }

    return y_oof_pred, residuals, metrics



def kfold_cv_welsch(X_eval, y_eval, best_tau, n_splits=5, seed=42, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    y_oof_pred = np.empty_like(y_eval, dtype=float)

    fold_medians = []
    fold_iqrs = []
    fold_maes = []

    for train_idx, test_idx in kf.split(X_eval):
        X_tr, y_tr = X_eval[train_idx], y_eval[train_idx]
        X_te, y_te = X_eval[test_idx], y_eval[test_idx]

        model = WeslchAlgo(X_tr, y_tr)
        beta_hat = model.optimizer_approach(
            method='BFGS',
            tau=best_tau,
            maxiter=100,
            initial_guess=np.zeros(X_tr.shape[1]),
            maxiter_first_stage=100,
        )

        y_pred_fold = X_te @ beta_hat
        y_oof_pred[test_idx] = y_pred_fold

        residuals_fold = y_te - y_pred_fold
        iqr = np.quantile(residuals_fold, 0.75) - np.quantile(residuals_fold, 0.25)

        fold_medians.append(np.median(np.abs(residuals_fold)))
        fold_iqrs.append(iqr)
        fold_maes.append(mean_absolute_error(y_te, y_pred_fold))

    residuals = y_eval - y_oof_pred

    metrics = {
        "Welsch Metrics": {
            "Median of MedAE": float(np.median(fold_medians)),
            "Median of MeanAE": float(np.median(fold_maes)),
            "Median of IQR": float(np.median(fold_iqrs)),
        }
    }

    return y_oof_pred, residuals, metrics



def kfold_cv_lad(X_eval, y_eval, n_splits=5, seed=42, shuffle=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    y_oof_pred = np.empty_like(y_eval, dtype=float)

    fold_medians = []
    fold_iqrs = []
    fold_maes = []

    for train_idx, test_idx in kf.split(X_eval):
        X_tr, y_tr = X_eval[train_idx], y_eval[train_idx]
        X_te, y_te = X_eval[test_idx], y_eval[test_idx]

        model = LeastAbsoluteDeviationAlgo(X_tr, y_tr)
        beta_hat, _ = model.optimizer_approach(
            method='L-BFGS-B',
            max_iter=100,
            initial_guess=np.zeros(X_tr.shape[1]),
        )

        y_pred_fold = X_te @ beta_hat
        y_oof_pred[test_idx] = y_pred_fold

        residuals_fold = y_te - y_pred_fold
        iqr = np.quantile(residuals_fold, 0.75) - np.quantile(residuals_fold, 0.25)

        fold_medians.append(np.median(np.abs(residuals_fold)))
        fold_iqrs.append(iqr)
        fold_maes.append(mean_absolute_error(y_te, y_pred_fold))

    residuals = y_eval - y_oof_pred

    metrics = {
        "LAD Metrics": {
            "Median of MedAE": float(np.median(fold_medians)),
            "Median of MeanAE": float(np.median(fold_maes)),
            "Median of IQR": float(np.median(fold_iqrs)),
        }
    }

    return y_oof_pred, residuals, metrics