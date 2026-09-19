import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

def learning_data_regressor(CSV_FILE, LEARNING_OUT, TARGET):
    df = pd.read_csv(CSV_FILE, sep=";", encoding="utf-8-sig")
    print(f"Nombre de trades : {len(df)}")

    feature_cols = [
        'side', 'restant', 'variation', 'spread',
        'imbalance', 'delta_imbalance', 'ratio_flow',
        'moyenne_ratio', 'slope', 'tendance', 'whale',
        'imbalance_flow', 'flow_momentum', 
        'price_momentum', 'time_pressure', 'stake_var',
        # ── Features enrichies ──
        'imb_x_tend', 'flow_strength',
        'momentum_score', 'rr_theorique',
        'flow_acceleration', 'imbalance_acceleration',
    ]

    df["side"] = df["side"].apply(
        lambda x: 1 if str(x).strip() == "BUY"
                  else 0 if str(x).strip() == "SELL"
                  else int(float(x)) if str(x).strip() in ["0", "1", "0.0", "1.0"]
                  else np.nan
    )

    masque = (
        (df["ask"] >= 0.32) & (df["ask"] <= 0.82) &
        (df["restant"] >= 30) & (df["restant"] <= 200)
    )
    df_train = df[masque].copy().reset_index(drop=True)
    print(f"Lignes valides : {len(df_train)}")

    if TARGET not in df.columns:
        print(f"❌ Colonne '{TARGET}' introuvable")
        return

    feature_cols = [f for f in feature_cols if f in df.columns]
    X = df_train[feature_cols].fillna(0)
    y = df_train[TARGET].astype(np.float32)

    print(f"Cible : {TARGET} | min={y.min():.3f} | max={y.max():.3f} | mean={y.mean():.3f}")

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, rmse_scores, r2_scores = [], [], []
    last_model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"Fold {fold} → MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
        last_model = model

    print(f"\n=== MOYENNES 5 FOLDS ===")
    print(f"MAE  : {np.mean(mae_scores):.4f} (±{np.std(mae_scores):.4f})")
    print(f"RMSE : {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    print(f"R²   : {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")

    importances = pd.Series(last_model.feature_importances_, index=feature_cols)
    print("\nTop Features :\n", importances.sort_values(ascending=False).head(10))

    joblib.dump(last_model, LEARNING_OUT)
    print(f"\n✅ Sauvegardé : {LEARNING_OUT}")

# Lancement
# learning_data_regressor("entrainement_xboost_V3.csv", "xgboost_regressor_dd.pkl", "benef_max")
# learning_data_regressor("entrainement_xboost_V3.csv", "xgboost_regressor_benef_max.pkl", "benef_max")