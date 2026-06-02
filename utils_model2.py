# ==========================================================
# BONE AGE COMPLETE PIPELINE
# ==========================================================

import cv2
import joblib
import pickle
import numpy as np
import pandas as pd

from ultralytics import YOLO
from scipy.ndimage import label
from xgboost import XGBRegressor

# ==========================================================
# MODEL PATHS
# ==========================================================

FINGER_MODEL_PATH = "/home/sailesh/projects/rsna_webapp/models/finger_segmentation.pt"

CARPAL_MODEL_PATH = "/home/sailesh/projects/rsna_webapp/models/carpal_segmentation.pt"

GE8_CLASSIFIER_PATH = "/home/sailesh/projects/rsna_webapp/models/ge8_classifier.pkl"

GE8_CLASSIFIER_IMPUTER = (
    "/home/sailesh/projects/rsna_webapp/models/ge8_classifier_imputer.pkl"
)

GE8_LABEL_ENCODER = (
    "/home/sailesh/projects/rsna_webapp/models/ge8_label_encoder.pkl"
)

GE8_REG_MODEL = (
    "/home/sailesh/projects/rsna_webapp/models/xgb_boneage_model.json"
)

GE8_PREPROCESS_PIPELINE = (
    "/home/sailesh/projects/rsna_webapp/models/preprocessing_pipeline.pkl"
)

LE8_CLASSIFIER_PATH = "/home/sailesh/projects/rsna_webapp/models/le8_classifier.pkl"

LE8_CLASSIFIER_IMPUTER = (
    "/home/sailesh/projects/rsna_webapp/models/le8_classifier_imputer.pkl"
)

LE8_REG_MODEL = (
    "/home/sailesh/projects/rsna_webapp/models/le_8_xgb_boneage_model.json"
)

LE8_PREPROCESS_PIPELINE = (
    "/home/sailesh/projects/rsna_webapp/models/le_8_preprocessing_pipeline.pkl"
)

NUM_BONES = 19


# ==========================================================
# LOAD YOLO MODELS
# ==========================================================

finger_model = YOLO(
    FINGER_MODEL_PATH
)

carpal_model = YOLO(
    CARPAL_MODEL_PATH
)


# ==========================================================
# LOAD GE8 CLASSIFIER
# ==========================================================

ge8_classifier = joblib.load(
    GE8_CLASSIFIER_PATH
)

ge8_classifier_imputer = joblib.load(
    GE8_CLASSIFIER_IMPUTER
)

ge8_label_encoder = joblib.load(
    GE8_LABEL_ENCODER
)

GE8_EXPECTED_COLUMNS = list(
    ge8_classifier_imputer.feature_names_in_
)


# ==========================================================
# LOAD LE8 CLASSIFIER
# ==========================================================

le8_classifier = joblib.load(
    LE8_CLASSIFIER_PATH
)

le8_classifier_imputer = joblib.load(
    LE8_CLASSIFIER_IMPUTER
)

LE8_EXPECTED_COLUMNS = list(
    le8_classifier_imputer.feature_names_in_
)


# ==========================================================
# LOAD GE8 REGRESSION
# ==========================================================

ge8_regressor = XGBRegressor()

ge8_regressor.load_model(
    GE8_REG_MODEL
)

with open(
    GE8_PREPROCESS_PIPELINE,
    "rb"
) as f:

    ge8_pipeline_assets = pickle.load(f)

ge8_reg_imputer = ge8_pipeline_assets[
    "imputer"
]

ge8_variance_selector = ge8_pipeline_assets[
    "variance_selector"
]

ge8_drop_cols = ge8_pipeline_assets[
    "dropped_correlated_columns"
]

ge8_mi_selector = ge8_pipeline_assets[
    "mi_selector"
]

GE8_REG_COLUMNS = list(
    ge8_reg_imputer.feature_names_in_
)


# ==========================================================
# LOAD LE8 REGRESSION
# ==========================================================

le8_pipeline_assets = joblib.load(
    LE8_PREPROCESS_PIPELINE
)

le8_regressor = XGBRegressor()

le8_regressor.load_model(
    LE8_REG_MODEL
)

le8_reg_imputer = le8_pipeline_assets[
    "imputer"
]

le8_scaler = le8_pipeline_assets[
    "scaler"
]

le8_variance_selector = le8_pipeline_assets[
    "variance_selector"
]

le8_drop_cols = le8_pipeline_assets[
    "dropped_correlated_columns"
]

le8_mi_selector = le8_pipeline_assets[
    "mi_selector"
]

LE8_REG_COLUMNS = list(
    le8_reg_imputer.feature_names_in_
)


# ==========================================================
# BONE FEATURE EXTRACTION
# ==========================================================

def extract_bone_features(mask):

    if np.sum(mask) < 20:
        return None

    labeled, num = label(mask)

    if num > 0:

        largest = (
            np.argmax(
                np.bincount(
                    labeled.flat
                )[1:]
            ) + 1
        )

        mask = (
            labeled == largest
        ).astype(np.uint8)

    ys, xs = np.where(mask > 0)

    if len(xs) < 10:
        return None

    pts = np.column_stack(
        (xs, ys)
    ).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    proj = (
        (
            (xs - x0) * vx
        ) +
        (
            (ys - y0) * vy
        )
    ).flatten()

    length = (
        proj.max() - proj.min()
    )

    p_min = proj.min()

    lower = p_min + 0.3 * length

    upper = p_min + 0.7 * length

    mid_idx = np.where(
        (
            proj >= lower
        ) &
        (
            proj <= upper
        )
    )[0]

    if len(mid_idx) < 10:
        return None

    dist = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5
    )

    sample_idx = np.linspace(
        0,
        len(mid_idx) - 1,
        7
    ).astype(int)

    widths = []

    for i in sample_idx:

        idx = mid_idx[i]

        widths.append(
            dist[
                ys[idx],
                xs[idx]
            ] * 2
        )

    width = np.median(widths)

    if width <= 0:
        return None

    ratio = length / width

    return (
        float(length),
        float(width),
        float(ratio)
    )


# ==========================================================
# FINGER FEATURES
# ==========================================================

def extract_finger_features(image_path):

    results = finger_model(
        image_path,
        verbose=False
    )[0]

    best_masks = {}

    if results.masks is not None:

        masks = (
            results.masks
            .data
            .cpu()
            .numpy()
        )

        classes = (
            results.boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confs = (
            results.boxes
            .conf
            .cpu()
            .numpy()
        )

        for i, cls_idx in enumerate(classes):

            if cls_idx < NUM_BONES:

                if (
                    cls_idx not in best_masks
                    or
                    confs[i] >
                    best_masks[cls_idx]["conf"]
                ):

                    best_masks[cls_idx] = {

                        "mask": masks[i],

                        "conf": confs[i]
                    }

    feature_dict = {}

    for cls_idx in range(NUM_BONES):

        if cls_idx in best_masks:

            mask = best_masks[
                cls_idx
            ]["mask"]

            h, w = results.orig_shape

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            binary_mask = (
                mask > 0.5
            ).astype(np.uint8)

            res_feat = extract_bone_features(
                binary_mask
            )

            if res_feat is None:

                length = 0
                width = 0
                ratio = 0

            else:

                (
                    length,
                    width,
                    ratio
                ) = res_feat

        else:

            length = 0
            width = 0
            ratio = 0

        feature_dict[
            f"bone_{cls_idx}_length"
        ] = length

        feature_dict[
            f"bone_{cls_idx}_width"
        ] = width

        feature_dict[
            f"bone_{cls_idx}_ratio"
        ] = ratio

    return feature_dict


# ==========================================================
# CARPAL FEATURES
# ==========================================================

def extract_carpal_features(image_path):

    results = carpal_model.predict(
        source=image_path,
        conf=0.2,
        imgsz=1024,
        verbose=False
    )[0]

    h, w = results.orig_shape

    total_pixels = h * w

    if results.masks is None:

        return {

            "carpal_count": 0,

            "carpal_total": 0,

            "carpal_mean": 0,

            "carpal_max": 0
        }

    areas = []

    for mask in (
        results.masks
        .data
        .cpu()
        .numpy()
    ):

        area = np.sum(mask > 0.5)

        if area > 100:

            rel_area = (
                area / total_pixels
            )

            areas.append(rel_area)

    areas = sorted(
        areas,
        reverse=True
    )[:7]

    N = len(areas)

    total = np.sum(areas)

    mean = (
        total / N
        if N > 0 else 0
    )

    max_area = (
        areas[0]
        if N > 0 else 0
    )

    return {

        "carpal_count": float(N),

        "carpal_total": float(total),

        "carpal_mean": float(mean),

        "carpal_max": float(max_area)
    }


# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess_features(
    feature_dict,
    gender
):

    df = pd.DataFrame(
        [feature_dict]
    )

    gender_value = (
        1.5 if gender else 1
    )

    df["gender"] = gender_value

    df = df.replace(
        [np.inf, -np.inf],
        0
    )

    df = df.fillna(0)

    length_cols = [

        c for c in df.columns

        if "length" in c.lower()
    ]

    width_cols = [

        c for c in df.columns

        if "width" in c.lower()
    ]

    df["mean_length"] = (

        df[length_cols]

        .replace(0, np.nan)

        .mean(axis=1)
    )

    df["mean_length"] = (
        df["mean_length"]
        .fillna(1)
    )

    for col in length_cols:

        df[col] = (
            df[col] /
            df["mean_length"]
        )

    for col in width_cols:

        df[col] = (
            df[col] /
            df["mean_length"]
        )

    df.drop(
        columns=["mean_length"],
        inplace=True
    )

    return df


# ==========================================================
# COMBINE FEATURES
# ==========================================================

def combine_features(
    geometric_df,
    dl_feature_dict=None
):

    if dl_feature_dict is None:
        dl_feature_dict = {}

    dl_df = pd.DataFrame(
        [dl_feature_dict]
    )

    duplicate_cols = [

        c for c in dl_df.columns

        if c in geometric_df.columns
    ]

    dl_df = dl_df.drop(
        columns=duplicate_cols,
        errors="ignore"
    )

    final_df = pd.concat(
        [
            geometric_df.reset_index(drop=True),
            dl_df.reset_index(drop=True)
        ],
        axis=1
    )

    final_df = final_df.loc[
        :,
        ~final_df.columns.duplicated()
    ]

    return final_df


# ==========================================================
# CREATE BOUNDS
# ==========================================================

def create_bounds(predicted_age):

    if predicted_age is None:

        lower_bound = 0.0

        upper_bound = 1.5

    elif predicted_age == 0:

        lower_bound = 0.0

        upper_bound = 1.5

    else:

        lower_bound = (
            predicted_age - 0.5
        )

        upper_bound = (
            predicted_age + 1.5
        )

    return {

        "lower_bound": lower_bound,

        "upper_bound": upper_bound
    }


# ==========================================================
# GE8 CLASSIFIER
# ==========================================================

def predict_ge8_age(input_df):

    X = input_df.copy()

    for col in GE8_EXPECTED_COLUMNS:

        if col not in X.columns:

            X[col] = 0

    X = X[GE8_EXPECTED_COLUMNS]

    for col in X.columns:

        if X[col].dtype != object:

            X[col] = X[col].replace(
                0,
                np.nan
            )

    X_imp = pd.DataFrame(

        ge8_classifier_imputer.transform(X),

        columns=GE8_EXPECTED_COLUMNS
    )

    pred = ge8_classifier.predict(
        X_imp
    )[0]

    pred = int(pred)

    age = (
        ge8_label_encoder
        .inverse_transform([pred])[0]
    )

    return int(age)


# ==========================================================
# LE8 CLASSIFIER
# ==========================================================

def predict_le8_age(input_df):

    X = input_df.copy()

    for col in LE8_EXPECTED_COLUMNS:

        if col not in X.columns:

            X[col] = 0

    X = X[LE8_EXPECTED_COLUMNS]

    for col in X.columns:

        if X[col].dtype != object:

            X[col] = X[col].replace(
                0,
                np.nan
            )

    X_imp = pd.DataFrame(

        le8_classifier_imputer.transform(X),

        columns=LE8_EXPECTED_COLUMNS
    )

    pred = le8_classifier.predict(
        X_imp
    )[0]

    return int(pred)


# ==========================================================
# GE8 REGRESSION
# ==========================================================

def predict_ge8_boneage(final_df):

    X = final_df.copy()

    for col in GE8_REG_COLUMNS:

        if col not in X.columns:

            X[col] = 0

    X = X[GE8_REG_COLUMNS]

    for col in X.columns:

        if X[col].dtype != object:

            X[col] = X[col].replace(
                0,
                np.nan
            )

    X_imp = pd.DataFrame(

        ge8_reg_imputer.transform(X),

        columns=X.columns
    )

    X_var = ge8_variance_selector.transform(
        X_imp
    )

    selected_columns = X_imp.columns[
        ge8_variance_selector.get_support()
    ]

    X_filtered = pd.DataFrame(
        X_var,
        columns=selected_columns
    )

    X_filtered.drop(
        columns=ge8_drop_cols,
        inplace=True,
        errors="ignore"
    )

    X_selected = ge8_mi_selector.transform(
        X_filtered
    )

    pred = ge8_regressor.predict(

        np.asarray(
            X_selected,
            dtype=np.float32
        )

    )[0]

    return float(pred)


# ==========================================================
# LE8 REGRESSION
# ==========================================================

def predict_le8_boneage(input_df):

    X = input_df.copy()

    for col in LE8_REG_COLUMNS:

        if col not in X.columns:

            X[col] = 0

    X = X[LE8_REG_COLUMNS]

    for col in X.columns:

        if X[col].dtype != object:

            X[col] = X[col].replace(
                0,
                np.nan
            )

    X_imp = le8_reg_imputer.transform(X)

    X_scaled = le8_scaler.transform(
        X_imp
    )

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    X_var = le8_variance_selector.transform(
        X_scaled
    )

    selected_columns = X_scaled.columns[
        le8_variance_selector.get_support()
    ]

    X_var = pd.DataFrame(
        X_var,
        columns=selected_columns
    )

    X_var.drop(
        columns=le8_drop_cols,
        inplace=True,
        errors="ignore"
    )

    X_final = le8_mi_selector.transform(
        X_var
    )

    prediction = le8_regressor.predict(

        np.asarray(
            X_final,
            dtype=np.float32
        )

    )[0]

    return round(
        float(prediction),
        2
    )


# ==========================================================
# MASTER PREDICTION FUNCTION
# ==========================================================

def model2(

    image_path,

    gender,

    age_group="above_8",

    dl_feature_dict=None
):

    # ======================================================
    # DL FEATURES
    # ======================================================

    if dl_feature_dict is None:

        dl_feature_dict = {}

    # ======================================================
    # FINGER FEATURES
    # ======================================================

    finger_features = (
        extract_finger_features(
            image_path
        )
    )

    # ======================================================
    # GE8 PIPELINE
    # ======================================================

    if age_group == "above_8":

        processed_df = preprocess_features(

            finger_features,

            gender
        )

        final_df = combine_features(

            processed_df,

            dl_feature_dict
        )

        predicted_age = predict_ge8_age(
            final_df
        )

        bounds_dict = create_bounds(
            predicted_age
        )

        bounds_df = pd.DataFrame(
            [bounds_dict]
        )

        final_df = pd.concat(
            [
                final_df.reset_index(drop=True),

                bounds_df.reset_index(drop=True)
            ],
            axis=1
        )

        predicted_boneage = (
            predict_ge8_boneage(
                final_df
            )
        )

    # ======================================================
    # LE8 PIPELINE
    # ======================================================

    elif age_group == "below_8":

        carpal_features = (
            extract_carpal_features(
                image_path
            )
        )

        combined_features = {}

        combined_features.update(
            finger_features
        )

        combined_features.update(
            carpal_features
        )

        processed_df = preprocess_features(

            combined_features,

            gender
        )

        final_df = combine_features(

            processed_df,

            dl_feature_dict
        )

        predicted_age = predict_le8_age(
            final_df
        )

        bounds_dict = create_bounds(
            predicted_age
        )

        bounds_df = pd.DataFrame(
            [bounds_dict]
        )

        final_df = pd.concat(
            [
                final_df.reset_index(drop=True),

                bounds_df.reset_index(drop=True)
            ],
            axis=1
        )

        predicted_boneage = (
            predict_le8_boneage(
                final_df
            )
        )

    else:

        raise ValueError(
            "age_group must be "
            "'above_8' or 'below_8'"
        )

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

    return {

        "predicted_boneage_months":

            round(
                predicted_boneage,
                2
            ),

        "predicted_boneage_years":

            round(
                predicted_boneage / 12,
                2
            ),

        "predicted_age_class":

            predicted_age,

        "lower_bound":

            bounds_dict["lower_bound"],

        "upper_bound":

            bounds_dict["upper_bound"]
    }