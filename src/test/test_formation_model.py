import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def formation_test_phase(cfg, logger):

    logger.logger.info("=== Formation TEST phase ===")

    logger.logger.info(f"Test CSV path: {cfg['formation_test_csv']}")

    df = pd.read_csv(cfg["formation_test_csv"])
    logger.logger.info(f"Loaded TEST CSV rows: {len(df)}")

    feats = df.groupby("image").apply(build_features).reset_index()

    feats_path = os.path.join(logger.run_dir, "formation_test_features.csv")
    feats.to_csv(feats_path, index=False)

    model = joblib.load(os.path.join(logger.run_dir, "formation_model.pkl"))
    le = joblib.load(os.path.join(logger.run_dir, "formation_label_encoder.pkl"))

    y_true = le.transform(feats["formation"])
    X = feats.drop(columns=["image","formation"])

    pred = model.predict(X)

    acc = accuracy_score(y_true, pred)

    logger.logger.info(f"Formation TEST accuracy: {acc:.4f}")
    logger.logger.info("\n" + classification_report(y_true, pred))

    cm = confusion_matrix(y_true, pred)
    np.savetxt(
        os.path.join(logger.run_dir, "formation_confusion_matrix.txt"),
        cm,
        fmt="%d"
    )

    pred_labels = le.inverse_transform(pred)

    out = feats[["image","formation"]].copy()
    out["predicted"] = pred_labels

    out.to_csv(
        os.path.join(logger.run_dir, "formation_test_predictions.csv"),
        index=False
    )

    logger.logger.info("Formation TEST phase complete")
