import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def build_features(group):

    feats = {}

    feats["formation"] = group["formation"].iloc[0]

    wr = group[group.position=="WR"]
    rb = group[group.position.isin(["RB","FB","SB","WB"])]
    te = group[group.position=="TE"]
    ol = group[group.position=="OLINE"]
    qb = group[group.position=="QB"]
    skill = group[~group.position.isin(["OLINE","BG"])]

    pos_names = ["WR","RB","FB","SB","WB","TE","QB","OLINE"]

    for p in pos_names:
        feats[f"count_{p}"] = (group.position==p).sum()

    feats["wr_left"] = (wr.lr_align=="LEFT").sum()
    feats["wr_right"] = (wr.lr_align=="RIGHT").sum()

    feats["te_left"] = (te.lr_align=="LEFT").sum()
    feats["te_right"] = (te.lr_align=="RIGHT").sum()

    feats["rb_left"] = (rb.lr_align=="LEFT").sum()
    feats["rb_right"] = (rb.lr_align=="RIGHT").sum()

    if len(ol) > 1:
        oline_width = ol.dx_from_oline.max() - ol.dx_from_oline.min()
        feats["oline_width"] = oline_width
        feats["oline_dx_std"] = ol.dx_from_oline.std()
    else:
        oline_width = 0
        feats["oline_width"] = 0
        feats["oline_dx_std"] = 0

    if len(wr) > 1:
        feats["wr_dx_std"] = wr.dx_from_oline.std()
        feats["wr_dx_range"] = wr.dx_from_oline.max() - wr.dx_from_oline.min()
    else:
        feats["wr_dx_std"] = 0
        feats["wr_dx_range"] = 0

    feats["wr_depth_mean"] = wr.dy_from_oline.mean() if len(wr)>0 else 0

    feats["wr_dx_std_norm"] = (
        feats["wr_dx_std"] / oline_width if oline_width>0 else 0
    )

    feats["rb_depth_mean"] = rb.dy_from_oline.mean() if len(rb)>0 else 0
    feats["rb_depth_std"] = rb.dy_from_oline.std() if len(rb)>1 else 0
    feats["rb_dx_std"] = rb.dx_from_oline.std() if len(rb)>1 else 0

    qb_depth = qb.dy_from_oline.mean() if len(qb)>0 else 0
    feats["qb_depth"] = qb_depth

    feats["skill_depth_std"] = (
        skill.dy_from_oline.std() if len(skill)>1 else 0
    )

    feats["max_skill_depth"] = (
        skill.dy_from_oline.max() if len(skill)>0 else 0
    )

    feats["empty_backfield"] = int(feats["count_RB"]
                                   + feats["count_FB"]
                                   + feats["count_SB"]
                                   + feats["count_WB"] == 0)

    feats["shotgun_flag"] = int(qb_depth > 4)
    feats["under_center_flag"] = int(qb_depth < 2)

    feats["trips_left_flag"] = int(feats["wr_left"] >= 3)
    feats["trips_right_flag"] = int(feats["wr_right"] >= 3)
    feats["twins_flag"] = int(
        feats["wr_left"] == 2 or feats["wr_right"] == 2
    )

    feats["bunch_flag"] = int(
        feats["wr_dx_std_norm"] < 0.12 and feats["count_WR"] >= 3
    )

    feats["double_te_flag"] = int(feats["count_TE"] >= 2)
    feats["te_same_side_flag"] = int(
        feats["te_left"] >= 2 or feats["te_right"] >= 2
    )

    feats["heavy_flag"] = int(
        feats["count_TE"] + feats["count_RB"] + feats["count_FB"]
        + feats["count_WB"] >= 4
    )

    feats["skill_left"] = (skill.lr_align=="LEFT").sum()
    feats["skill_right"] = (skill.lr_align=="RIGHT").sum()

    feats["lr_balance_diff"] = abs(
        feats["skill_left"] - feats["skill_right"]
    )

    if len(skill)>1:
        x_range = skill.dx_from_oline.max() - skill.dx_from_oline.min()
        y_range = skill.dy_from_oline.max() - skill.dy_from_oline.min()
    else:
        x_range = 0
        y_range = 0

    feats["skill_x_range"] = x_range
    feats["skill_y_range"] = y_range
    feats["skill_area"] = x_range * y_range

    return pd.Series(feats)


def formation_train_phase(cfg, logger):

    logger.logger.info("Starting formation classification TRAIN phase")

    df = pd.read_csv(cfg["formation_train_csv"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.logger.info(f"Loaded players CSV: {len(df)} rows")

    feats_df = df.groupby("image").apply(build_features).reset_index()

    features_path = os.path.join(logger.run_dir, "formation_train_features.csv")
    feats_df.to_csv(features_path, index=False)

    logger.logger.info(f"Saved features → {features_path}")
    logger.logger.info(f"Formations: {feats_df['formation'].value_counts().to_dict()}")

    le = LabelEncoder()
    y = le.fit_transform(feats_df["formation"])
    X = feats_df.drop(columns=["image","formation"])

    le_path = os.path.join(logger.run_dir, "formation_label_encoder.pkl")
    joblib.dump(le, le_path)
    logger.logger.info(f"Saved label encoder → {le_path}")

    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=cfg.get("num_workers", 8),
        random_state=cfg["seed"]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])

    scores = []
    best_acc = 0
    best_model = None
    best_fold = 0

    logger.logger.info("Running 5-fold cross validation")

    for fold,(tr,va) in enumerate(cv.split(X,y),1):

        fold_model = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=cfg.get("num_workers", 8),
            random_state=cfg["seed"]
        )

        fold_model.fit(X.iloc[tr], y[tr])
        pred = fold_model.predict(X.iloc[va])

        acc = accuracy_score(y[va], pred)
        scores.append(acc)

        logger.logger.info(f"Fold {fold} accuracy: {acc:.4f}")
        logger.logger.info("\n" + classification_report(y[va], pred))

        if acc > best_acc:
            best_acc = acc
            best_model = fold_model
            best_fold = fold

    logger.logger.info(f"CV mean accuracy: {np.mean(scores):.4f}")
    logger.logger.info(f"Best fold: {best_fold} acc={best_acc:.4f}")

    model_path = os.path.join(logger.run_dir, "formation_model.pkl")
    joblib.dump(best_model, model_path)
    logger.logger.info(f"Saved formation model → {model_path}")
