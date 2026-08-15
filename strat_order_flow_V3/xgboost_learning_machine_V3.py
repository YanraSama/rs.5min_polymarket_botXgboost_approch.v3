import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def learning_data_classifier(CSV_FILE, LEARNING_OUT):

    # ====================== CHARGEMENT ======================
    df = pd.read_csv(CSV_FILE, sep=";", encoding="utf-8-sig", low_memory=False)
    print(f"Nombre de trades : {len(df)}")
    print("Distribution target :\n", df["target"].value_counts())

    # ====================== CONVERSION SIDE ======================
    def convert_side(x):
        x = str(x).strip()
        if x == "BUY":  return 1
        if x == "SELL": return 0
        try:            return int(float(x))
        except:         return np.nan

    df["side"] = df["side"].apply(convert_side)

    for col in df.columns:
        if col != "side":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ====================== FEATURES ENRICHIES ======================
    # Interactions pertinentes basées sur l'analyse précédente
    # df["imb_x_ratio"]    = df["imbalance"]  * df["ratio_flow"]
    # df["imb_x_tend"]     = df["imbalance"]  * df["tendance"]
    # df["flow_strength"]  = df["ratio_flow"].abs() * df["moyenne_ratio"].abs()
    # df["momentum_score"] = df["ratio_flow"] * df["moyenne_ratio"] * df["tendance"]
    # df["time_ratio"]     = df["restant"] / 300.0
    # df["rr_theorique"]   = (1.0 - df["ask"]) / df["ask"].clip(0.01)

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

    # ====================== FILTRE QUALITÉ ======================
    # masque_qualite = (
        # (df["ask"]       >= 0.32) &
        # (df["ask"]       <= 0.82) &   # ← réduit de 0.92 à 0.82
        # (df["imbalance"] >= 0.05) &   # ← ajouté
        # (df["restant"]   >= 30)   &
        # (df["restant"]   <= 200)      # ← ajouté
    # )

    # print(f"\nLignes valides  : {masque_qualite.sum()} ({masque_qualite.mean()*100:.1f}%)")
    # print(f"Lignes ignorées : {(~masque_qualite).sum()} ({(~masque_qualite).mean()*100:.1f}%)")

    # if masque_qualite.sum() == 0:
        # print("Datas empty")
        # return

    # ── Supprimer les NaN du target ──
    df_train = df.dropna(subset=["target"]).reset_index(drop=True)
    df_train["target"] = df_train["target"].astype(int)  # 0 / 1 strict


    # df_train = df[masque_qualite].copy().reset_index(drop=True)
    print(f"\nDataset d'entraînement : {len(df_train)} lignes")
    print("Distribution target :\n", df_train["target"].value_counts())

    # Garder uniquement les features présentes
    feature_cols = [f for f in feature_cols if f in df_train.columns]
    print(f"\nFeatures utilisées ({len(feature_cols)}) : {feature_cols}")

    X = df_train[feature_cols].fillna(0)
    y = df_train["target"]

    # ====================== VALIDATION CROISÉE CHRONOLOGIQUE ======================
    tscv      = TimeSeriesSplit(n_splits=5)
    auc_scores = []
    last_model, last_y_test, last_proba = None, None, None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]       
        
    # split_index = int(len(df_train) * 0.8)        

    # train_df = df_train.iloc[:split_index]
    # test_df = df_train.iloc[split_index:]
    
    # X_train = train_df[feature_cols].fillna(0)
    # y_train = train_df["target"]
    # X_test = test_df[feature_cols].fillna(0)
    # y_test = test_df["target"]
    
    # print(f"\n--- Fold {fold} | Train={len(X_train)} | Test={len(X_test)} ---")

        ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators          = 1000,
            learning_rate         = 0.03,
            max_depth             = 5,
            min_child_weight      = 5,
            subsample             = 0.75,
            colsample_bytree      = 0.70,
            colsample_bylevel     = 0.70,
            colsample_bynode      = 0.70,
            reg_alpha             = 0.5,
            reg_lambda            = 1.0,
            scale_pos_weight      = ratio,
            tree_method           = "hist",
            eval_metric           = "logloss",  # ← garder logloss
            early_stopping_rounds = 30,
            random_state          = 42,
            n_jobs                = -1,
        )

        model.fit(
            X_train, y_train,
            eval_set = [(X_test, y_test)],
            verbose  = False,
        )

        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)
        auc_scores.append(auc)
        # print(f"Fold {fold} AUC : {auc:.4f}")

        last_model  = model
        last_y_test = y_test
        last_proba  = proba

    print(f"\n=== AUC moyen : {np.mean(auc_scores):.4f} (± {np.std(auc_scores):.4f}) ===")

    # ====================== ANALYSE DU SEUIL ======================
    print("\n=== WR RÉEL PAR SEUIL DE PROBA ===")
    for seuil in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = last_proba >= seuil
        n1   = mask.sum()
        if n1 == 0:
            continue
        wr   = last_y_test[mask].mean() * 100
        pct  = n1 / len(last_proba) * 100
        acc  = accuracy_score(last_y_test, (last_proba >= seuil).astype(int))

        # Faux positifs et vrais positifs
        tp   = ((last_proba >= seuil) & (last_y_test == 1)).sum()
        fp   = ((last_proba >= seuil) & (last_y_test == 0)).sum()

        print(f"Seuil {seuil:.2f} | N={n1:6d} ({pct:4.1f}%) | "
              f"WR={wr:5.1f}% | TP={tp:5d} | FP={fp:5d} | Acc={acc:.3f}")

    # ====================== RÉSULTATS FINAUX ======================
    seuil_final = 0.50
    pred_final  = (last_proba >= seuil_final).astype(int)

    print(f"\n=== RÉSULTATS FINAUX (seuil={seuil_final}) ===")
    print(f"Accuracy : {accuracy_score(last_y_test, pred_final):.4f}")
    print(f"AUC      : {roc_auc_score(last_y_test, last_proba):.4f}")
    print(classification_report(last_y_test, pred_final, zero_division=0))
    print("Matrice de confusion :\n", confusion_matrix(last_y_test, pred_final))

    # ====================== FEATURE IMPORTANCE ======================
    importances = pd.Series(
        last_model.feature_importances_,
        index = feature_cols
    ).sort_values(ascending=False)

    print("\nTop Features :")
    print(importances.head(15))

    # ── Alerte si une feature domine trop ──
    top_feat_pct = importances.iloc[0]
    if top_feat_pct > 0.25:
        print(f"\n⚠️ '{importances.index[0]}' domine à {top_feat_pct*100:.1f}%")
        print("   → Risque de surapprentissage sur cette feature")

    # ====================== SAUVEGARDE ======================
    joblib.dump(last_model,  LEARNING_OUT)
    joblib.dump(feature_cols, LEARNING_OUT.replace(".pkl", "_features.pkl"))
    print(f"\n✅ Modèle   sauvegardé : {LEARNING_OUT}")
    print(f"✅ Features sauvegardées : {LEARNING_OUT.replace('.pkl', '_features.pkl')}")
    
    return accuracy_score(last_y_test, pred_final), roc_auc_score(last_y_test, last_proba), confusion_matrix(last_y_test, pred_final), importances
    

# learning_data_classifier("entrainement_xboost_V3.csv", "xgboost_model_v3_target.pkl")