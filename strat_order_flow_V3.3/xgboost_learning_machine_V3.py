import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

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
# ============================================================
# OPTIMISATION SCALE_POS_WEIGHT
# ============================================================

    SCALE_FACTORS = [
        0.50,
        0.75,
        1.00,
        1.10,
        1.25,
        1.50,
        1.75,
        2.00,
    ]

    THRESHOLDS = [
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    tscv = TimeSeriesSplit(n_splits=5)

    results = []

    best_score = -np.inf
    best_model = None
    best_factor = None
    best_weight = None
    best_proba = None
    best_y_test = None
    best_feature_cols = feature_cols

    super_wr = 0
    super_seuil = 0

    # ============================================================
    # BOUCLE SCALE_POS_WEIGHT
    # ============================================================

    for factor in SCALE_FACTORS:

        print("\n" + "=" * 70)
        print(f"SCALE FACTOR = {factor:.2f}")
        print("=" * 70)

        fold_aucs = []
        fold_wrs = {}
        fold_ns = {}

        current_last_model = None
        current_last_proba = None
        current_last_y_test = None
        
        average_cis_1 = 0
        average_cis_0 = 0
        average_recall_1 = 0
        average_recall_0 = 0

        for fold, (train_idx, test_idx) in enumerate(
            tscv.split(X), start=1
        ):

            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]
            
            # ----------------------------------------------------
            # Vérification des classes
            # ----------------------------------------------------

            train_classes = y_train.nunique()
            test_classes  = y_test.nunique()

            if train_classes < 2:
                print(
                    f"⚠️ Fold {fold} ignoré : "
                    f"train = {y_train.value_counts().to_dict()}"
                )
                continue

            if test_classes < 2:
                print(
                    f"⚠️ Fold {fold} ignoré : "
                    f"test = {y_test.value_counts().to_dict()}"
                )
                continue

            # ----------------------------------------------------
            # Ratio naturel
            # ----------------------------------------------------

            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()

            ratio = n_negative / max(n_positive, 1)

            scale_pos_weight = ratio * factor

            print(
                f"Fold {fold} | "
                f"0={n_negative:,} | "
                f"1={n_positive:,} | "
                f"ratio={ratio:.3f} | "
                f"SPW={scale_pos_weight:.3f}"
            )

            # ----------------------------------------------------
            # MODELE
            # ----------------------------------------------------

            model = xgb.XGBClassifier(

                n_estimators=1000,

                learning_rate=0.03,

                max_depth=3,

                min_child_weight=5,

                subsample=0.75,

                colsample_bytree=0.70,
                colsample_bylevel=0.70,
                colsample_bynode=0.70,

                reg_alpha=1.0,
                reg_lambda=2.0,

                scale_pos_weight=scale_pos_weight,

                tree_method="hist",

                eval_metric="auc",

                early_stopping_rounds=30,

                random_state=42,

                n_jobs=-1,
            )

            # ----------------------------------------------------
            # TRAIN
            # ----------------------------------------------------

            model.fit(
                X_train,
                y_train,

                eval_set=[
                    (X_test, y_test)
                ],

                verbose=False,
            )

            # ----------------------------------------------------
            # PROBA
            # ----------------------------------------------------

            proba = model.predict_proba(
                X_test
            )[:, 1]

            auc = roc_auc_score(
                y_test,
                proba
            )

            fold_aucs.append(auc)

            # ----------------------------------------------------
            # WR PAR SEUIL
            # ----------------------------------------------------

            print(
                f"Fold {fold} | "
                f"AUC={auc:.4f}"
            )

            for threshold in THRESHOLDS:

                mask = proba >= threshold

                n = mask.sum()

                if n == 0:
                    continue

                wr = y_test[mask].mean()

                fold_wrs.setdefault(
                    threshold,
                    []
                ).append(wr)

                fold_ns.setdefault(
                    threshold,
                    []
                ).append(n)

            current_last_model = model
            current_last_proba = proba
            current_last_y_test = y_test
            
            seuil = 0.65
            pred = (proba >= seuil).astype(int)

            # Precision sur la classe 1 (les trades pris)
            average_cis_1 += precision_score(y_test, pred, pos_label=1, zero_division=0)

            # Precision sur la classe 0
            average_cis_0 += precision_score(y_test, pred, pos_label=0, zero_division=0)
            
            # Precision sur la classe 1 (les trades pris)
            average_recall_1 += recall_score(y_test, pred, pos_label=1, zero_division=0)

            # Precision sur la classe 0
            average_recall_0 += recall_score(y_test, pred, pos_label=0, zero_division=0)
            

            print(f"Precision(1) = {average_cis_1:.4f}", "  ", precision_score(y_test, pred, pos_label=1, zero_division=0), "   ", recall_score(y_test, pred, pos_label=1, zero_division=0))
            print(f"Precision(0) = {average_cis_0:.4f}", "  ", precision_score(y_test, pred, pos_label=0, zero_division=0), "   ", recall_score(y_test, pred, pos_label=0, zero_division=0))                
            
        
        # ========================================================
        # RESULTATS DU SCALE
        # ========================================================

        if len(fold_aucs) == 0:

            print(
                f"❌ Aucun fold valide pour factor={factor}"
            )

            continue

        mean_auc = np.mean(fold_aucs)
        std_auc  = np.std(fold_aucs)

        print(
            f"\nSCALE FACTOR {factor:.2f}"
        )

        print(
            f"AUC moyen = {mean_auc:.4f} "
            f"(± {std_auc:.4f})"
        )

        # --------------------------------------------------------
        # WR MOYEN PAR SEUIL
        # --------------------------------------------------------

        best_threshold = None
        best_wr = -np.inf
        best_n = 0

        for threshold in THRESHOLDS:

            if threshold not in fold_wrs:
                continue

            mean_wr = np.mean(
                fold_wrs[threshold]
            )

            mean_n = np.mean(
                fold_ns[threshold]
            )

            print(
                f"Seuil {threshold:.2f} | "
                f"WR moyen={mean_wr*100:.2f}% | "
                f"N moyen={mean_n:.0f}"
            )

            # Pour l'instant :
            # objectif = WR
            
            # print ("check SNT : ", mean_wr, "   ", best_wr, "   ", mean_n, "    ", mean_wr > best_wr and mean_n > 500, "    ", threshold, "   ", best_threshold)
            if mean_wr > best_wr and mean_n > 500:

                best_wr = mean_wr
                best_threshold = threshold
                best_n = mean_n
                
        # --------------------------------------------------------
        # SCORE GLOBAL
        # --------------------------------------------------------

        # IMPORTANT :
        # on ne choisit PAS uniquement le WR.
        #
        # Ici je donne priorité à l'AUC,
        # puis au WR à un seuil donné.

        score = mean_auc

        results.append({

            "factor": factor,

            "mean_auc": mean_auc,

            "std_auc": std_auc,

            "best_threshold": best_threshold,

            "best_wr": best_wr,

            "mean_n": best_n,

            "scale_pos_weight_last":
                ratio * factor,

        })
                
        # --------------------------------------------------------
        # MEILLEUR MODELE
        # --------------------------------------------------------
        
        # print ("test infos : ", best_n > 500, " " , best_threshold >= 0.60, "   ", best_wr > super_wr, "    ", average_cis_1/5 > 0.50, "    ", average_cis_0/5 > 0.30, "    ", average_recall_1/5 > 0.3, "      ", average_recall_0/5 > 0.3, "      ", average_recall_1/5, "   ", average_recall_0/5, "    ", average_cis_1/5, "  ", average_cis_0/5, "   ", factor, "   ", best_n, "    ", best_threshold, "    ", best_wr, "   ", super_wr, "  ", mean_n > 500 and best_threshold >= 0.60 and best_wr > super_wr)

        if best_n > 500 and best_threshold >= 0.60 and best_wr > super_wr and average_cis_1/5 > 0.50 and  average_cis_0/5 > 0.30 and average_recall_1/5 > 0.3 and average_recall_0/5 > 0.3:

            best_score = score

            best_factor = factor

            best_model = current_last_model

            best_weight = ratio * factor

            best_proba = current_last_proba

            best_y_test = current_last_y_test
            
            super_wr = best_wr
            
            super_seuil = best_threshold


    # ============================================================
    # RESULTATS FINAUX
    # ============================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "mean_auc",
        ascending=False
    )
    
    # ============================================================
    # CALIBRATION + ÉVALUATION FINALE PROPRE
    # ============================================================

    print("\n" + "="*80)
    print("CALIBRATION DES PROBABILITÉS + ANALYSE RÉELLE")
    print("="*80)

    # On reprend le meilleur modèle trouvé
    # (best_model, best_proba, best_y_test viennent de ton code actuel)

    # ---------- 1. Distribution des probabilités brutes ----------
    print("\nDistribution des probabilités (dernier fold) :")
    print(pd.Series(best_proba).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        
    # fallback simple
    if current_last_model is None:
        print("❌ Impossible : aucun modèle n'a été entraîné.")
        return 0, 0, None, None, None, 0.90    

    print("\nNombre d'opportunités selon le seuil (probabilités brutes) :")
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        n = (best_proba >= th).sum()
        if n > 0:
            wr = best_y_test[best_proba >= th].mean()
            print(f"Seuil {th:.2f} → N={n:5d} | WR={wr*100:.2f}%")
        else:
            print(f"Seuil {th:.2f} → N=    0")

    # ---------- 2. Calibration (Isotonic) ----------
    print("\nCalibration des probabilités (Isotonic Regression)...")

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(best_proba, best_y_test)

    proba_calibrated = iso.predict(best_proba)

    print("\nDistribution après calibration :")
    print(pd.Series(proba_calibrated).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\nNombre d'opportunités après calibration :")
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        n = (proba_calibrated >= th).sum()
        if n > 0:
            wr = best_y_test[proba_calibrated >= th].mean()
            print(f"Seuil {th:.2f} → N={n:5d} | WR={wr*100:.2f}%")
        else:
            print(f"Seuil {th:.2f} → N=    0")

    # ---------- 3. Choix intelligent du seuil ----------
    print("\n" + "="*60)
    print("RECHERCHE DU MEILLEUR SEUIL (après calibration)")
    print("="*60)

    best_th = None
    best_wr = 0
    best_n = 0

    MIN_N = 300          # minimum d'opportunités souhaité
    MIN_WR = 0.60        # WR minimum acceptable

    for th in np.arange(0.50, 0.85, 0.01):
        mask = proba_calibrated >= th
        n = mask.sum()
        if n < MIN_N:
            continue
        wr = best_y_test[mask].mean()
        if wr >= MIN_WR and (wr > best_wr or (wr == best_wr and n > best_n)) and th <= 0.75:
            best_wr = wr
            best_th = th
            best_n = n

    if best_th is not None:
        print(f"\n✅ Meilleur seuil trouvé : {best_th:.2f}")
        print(f"   → WR = {best_wr*100:.2f}%")
        print(f"   → N  = {best_n}")
    else:
        print("\n⚠️ Aucun seuil ne respecte tes critères (MIN_N + MIN_WR)")
        # fallback
        best_th = 0.65
        print(f"   → Seuil de secours utilisé : {best_th}")

    # ---------- 4. Sauvegarde ----------
    # On sauvegarde aussi le calibrateur
    joblib.dump(best_model, LEARNING_OUT)
    joblib.dump(feature_cols, LEARNING_OUT.replace(".pkl", "_features.pkl"))
    joblib.dump(iso, LEARNING_OUT.replace(".pkl", "_calibrator.pkl"))

    print(f"\n✅ Modèle sauvegardé          : {LEARNING_OUT}")
    print(f"✅ Features sauvegardées      : {LEARNING_OUT.replace('.pkl', '_features.pkl')}")
    print(f"✅ Calibrateur sauvegardé     : {LEARNING_OUT.replace('.pkl', '_calibrator.pkl')}")
    print(f"✅ Seuil recommandé           : {best_th:.2f}")    
    
   
    # print("\n")
    # print("=" * 80)
    # print("CLASSEMENT SCALE_POS_WEIGHT")
    # print("=" * 80)

    # print(
        # results_df.to_string(
            # index=False
        # )
    # )

    # if best_factor == None:
        # print ("ALL SET IS INCONSISTENT...")
        # return 0, 0, 0, 0, 0, 0
    
    # print("\n🏆 MEILLEUR SCALE FACTOR")
    # print(
        # f"Factor : {best_factor:.2f}", "    Seuil : ",  super_seuil
    # )

    # print(
        # f"Scale_pos_weight : {best_weight:.4f}"
    # )

    # print(
        # f"AUC moyen : {best_score:.4f}"
    # )    
    
    pred_final  = (best_proba >= 0.65).astype(int)

    # print(f"Accuracy : {accuracy_score(best_y_test, pred_final):.4f}")
    # print(f"AUC      : {roc_auc_score(best_y_test, best_proba):.4f}")
    # print(classification_report(best_y_test, pred_final, zero_division=0))
    # print("Matrice de confusion :\n", confusion_matrix(best_y_test, pred_final))

      # ====================== FEATURE IMPORTANCE ======================
    importances = pd.Series(
        best_model.feature_importances_,
        index = feature_cols
    ).sort_values(ascending=False)

    # print("\nTop Features :")
    # print(importances.head(15))
    
    # ====================== SAUVEGARDE ======================
    # joblib.dump(best_model,  LEARNING_OUT)
    # joblib.dump(feature_cols, LEARNING_OUT.replace(".pkl", "_features.pkl"))
    # print(f"\n✅ Modèle   sauvegardé : {LEARNING_OUT}")
    # print(f"✅ Features sauvegardées : {LEARNING_OUT.replace('.pkl', '_features.pkl')}")
    
    print ("meilleur seuil : ", best_th)
    return accuracy_score(best_y_test, pred_final), roc_auc_score(best_y_test, best_proba), confusion_matrix(best_y_test, pred_final), importances, classification_report(best_y_test, pred_final, zero_division=0), best_th

    
# learning_data_classifier("learning_live_v3.csv", "xgboost_model_v3_target.pkl")