import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix

from statsmodels.miscmodels.ordinal_model import OrderedModel


warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths
# ============================================================

DATA_DIR = "/Users/robertchen/Desktop/UChicago Spring 2026/BUSN 20800/Final Project (Auth)/Data"

INPUT_FILE = os.path.join(
    DATA_DIR,
    "philadelphia_no_review_category_attribute_dummy_features_with_review_count.csv"
)

OUTPUT_DIR = os.path.join(DATA_DIR, "ordered_logit_outputs_3group_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

if "stars" not in df.columns:
    raise ValueError("Dataset must contain a stars column.")

data = df.dropna(subset=["stars"]).copy()

print("Loaded data shape:", data.shape)
print("\nOriginal stars distribution:")
print(data["stars"].value_counts().sort_index())


# ============================================================
# 3. Create 3 ordered rating groups
# ============================================================
# Group 1: 1.0–3.0      Low / average-or-below
# Group 2: 3.5–4.0      Good
# Group 3: 4.5–5.0      Excellent

data["rating_group"] = np.select(
    [
        data["stars"] <= 3.0,
        data["stars"] <= 4.0,
        data["stars"] > 4.0,
    ],
    [1, 2, 3],
)

data["rating_group_code"] = data["rating_group"] - 1

group_labels = {
    1: "low_or_average_below",
    2: "good_3_5_to_4_0",
    3: "excellent_4_5_to_5_0",
}

group_to_midstar = {
    1: 2.50,
    2: 3.75,
    3: 4.75,
}

print("\nThree-group rating distribution:")
print(data["rating_group"].value_counts().sort_index())

print("\nThree-group rating share:")
print(data["rating_group"].value_counts(normalize=True).sort_index())


# ============================================================
# 4. Feature engineering
# ============================================================

for col in ["review_count", "checkin_count", "tip_count", "photo_count"]:
    if col in data.columns:
        data[f"log_{col}"] = np.log1p(data[col])


data["price_range"] = np.nan

price_dummy_map = {
    "attr_restaurantspricerange2_1": 1,
    "attr_restaurantspricerange2_2": 2,
    "attr_restaurantspricerange2_3": 3,
    "attr_restaurantspricerange2_4": 4,
}

for dummy_col, price_value in price_dummy_map.items():
    if dummy_col in data.columns:
        data.loc[data[dummy_col] == 1, "price_range"] = price_value


# ============================================================
# 5. Predictor set
# ============================================================

candidate_predictors = [
    "log_review_count",
    "log_checkin_count",
    "log_tip_count",
    "log_photo_count",

    "latitude",
    "longitude",

    "is_open",
    "price_range",

    "has_takeout",
    "has_delivery",

    "attr_outdoorseating",
    "attr_bikeparking",
    "attr_restaurantsgoodforgroups",
    "attr_restaurantsreservations",
    "attr_goodforkids",
    "attr_businessacceptscreditcards",

    "attr_noiselevel_average",
    "attr_noiselevel_quiet",
]

predictors = [col for col in candidate_predictors if col in data.columns]

print("\nPredictors used:")
for col in predictors:
    print("-", col)


# ============================================================
# 6. Build modeling data
# ============================================================

model_df = data[["stars", "rating_group", "rating_group_code"] + predictors].copy()

for col in predictors:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

usable_predictors = []

for col in predictors:
    if model_df[col].isna().all():
        continue

    if model_df[col].nunique(dropna=True) <= 1:
        continue

    model_df[col] = model_df[col].fillna(model_df[col].median())
    usable_predictors.append(col)

predictors = usable_predictors

model_df = model_df[["stars", "rating_group", "rating_group_code"] + predictors].copy()
model_df["rating_group_code"] = model_df["rating_group_code"].astype(int)

print("\nModeling data shape:", model_df.shape)
print("Number of predictors:", len(predictors))


# ============================================================
# 7. Train-test split
# ============================================================

X = model_df[predictors].copy()
y = model_df["rating_group_code"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# ============================================================
# 8. Standardize continuous predictors
# ============================================================

continuous_predictors = [
    "log_review_count",
    "log_checkin_count",
    "log_tip_count",
    "log_photo_count",
    "latitude",
    "longitude",
    "price_range",
]

continuous_predictors = [col for col in continuous_predictors if col in predictors]

X_train_model = X_train.copy()
X_test_model = X_test.copy()

scaling_rows = []

for col in continuous_predictors:
    mean = X_train[col].mean()
    std = X_train[col].std()

    if pd.isna(std) or std == 0:
        std = 1

    X_train_model[col] = (X_train[col] - mean) / std
    X_test_model[col] = (X_test[col] - mean) / std

    scaling_rows.append(
        {
            "variable": col,
            "train_mean": mean,
            "train_std": std,
        }
    )

pd.DataFrame(scaling_rows).to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_scaling_parameters.csv"),
    index=False,
)


# ============================================================
# 9. Fit ordered logit
# ============================================================
# Important: do NOT add a constant.

model = OrderedModel(
    endog=y_train,
    exog=X_train_model,
    distr="logit",
)

result = model.fit(
    method="bfgs",
    maxiter=1000,
    disp=True,
)

print("\nOrdered logit fitted successfully.")
print(result.summary())

with open(os.path.join(OUTPUT_DIR, "ordered_logit_3group_summary.txt"), "w") as f:
    f.write(str(result.summary()))


# ============================================================
# 10. Coefficients and odds ratios
# ============================================================

coef_df = pd.DataFrame(
    {
        "term": result.params.index,
        "coefficient": result.params.values,
        "std_error": result.bse.values,
        "p_value": result.pvalues.values,
        "odds_ratio": np.exp(result.params.values),
    }
)

coef_df["is_predictor"] = coef_df["term"].isin(predictors)

coef_df.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_coefficients_and_odds_ratios.csv"),
    index=False,
)

predictor_coef_df = coef_df[coef_df["is_predictor"]].copy()

predictor_coef_df["direction"] = np.where(
    predictor_coef_df["coefficient"] > 0,
    "higher_rating_group",
    "lower_rating_group",
)

predictor_coef_df["significant_5pct"] = predictor_coef_df["p_value"] < 0.05

predictor_coef_df = predictor_coef_df.sort_values("p_value")

predictor_coef_df.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_predictor_coefficients.csv"),
    index=False,
)


# ============================================================
# 11. Predict using argmax only
# ============================================================

pred_prob = np.asarray(result.model.predict(result.params, exog=X_test_model))

actual_code = y_test.values
actual_group = actual_code + 1

pred_code = pred_prob.argmax(axis=1)
pred_group = pred_code + 1

actual_star_approx = np.array([group_to_midstar[g] for g in actual_group])
pred_star_approx = np.array([group_to_midstar[g] for g in pred_group])


# ============================================================
# 12. Metrics
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "model": "ordered_logit_3group",
            "prediction_rule": "argmax",
            "num_observations": len(model_df),
            "num_train": len(X_train),
            "num_test": len(X_test),
            "num_predictors": len(predictors),
            "accuracy_exact_group": accuracy_score(actual_code, pred_code),
            "mae_group": mean_absolute_error(actual_group, pred_group),
            "rmse_group": np.sqrt(mean_squared_error(actual_group, pred_group)),
            "within_one_group_accuracy": np.mean(np.abs(pred_group - actual_group) <= 1),
            "mae_star_approx": mean_absolute_error(actual_star_approx, pred_star_approx),
            "rmse_star_approx": np.sqrt(mean_squared_error(actual_star_approx, pred_star_approx)),
        }
    ]
)

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_metrics.csv"),
    index=False,
)

print("\nMetrics:")
print(metrics_df)


# ============================================================
# 13. Confusion matrix
# ============================================================

row_names = [
    "actual_1_low_or_average_below",
    "actual_2_good_3_5_to_4_0",
    "actual_3_excellent_4_5_to_5_0",
]

col_names = [
    "pred_1_low_or_average_below",
    "pred_2_good_3_5_to_4_0",
    "pred_3_excellent_4_5_to_5_0",
]

cm_df = pd.DataFrame(
    confusion_matrix(actual_code, pred_code, labels=[0, 1, 2]),
    index=row_names,
    columns=col_names,
)

cm_df.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_confusion_matrix.csv")
)


# ============================================================
# 14. Prediction distribution table
# ============================================================

distribution_df = pd.DataFrame(
    {
        "group": [1, 2, 3],
        "label": [group_labels[1], group_labels[2], group_labels[3]],
        "actual_count": [np.sum(actual_group == g) for g in [1, 2, 3]],
        "predicted_count": [np.sum(pred_group == g) for g in [1, 2, 3]],
    }
)

distribution_df["actual_share"] = distribution_df["actual_count"] / len(actual_group)
distribution_df["predicted_share"] = distribution_df["predicted_count"] / len(pred_group)

distribution_df.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_prediction_distribution.csv"),
    index=False,
)


# ============================================================
# 15. Average predicted probabilities by actual group
# ============================================================

prob_df = pd.DataFrame(
    {
        "actual_group": actual_group,
        "prob_low_or_average_below": pred_prob[:, 0],
        "prob_good_3_5_to_4_0": pred_prob[:, 1],
        "prob_excellent_4_5_to_5_0": pred_prob[:, 2],
    }
)

avg_prob_by_actual_group = (
    prob_df
    .groupby("actual_group")
    .mean()
    .reset_index()
)

avg_prob_by_actual_group["actual_group_label"] = avg_prob_by_actual_group["actual_group"].map(group_labels)

avg_prob_by_actual_group.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_avg_predicted_prob_by_actual_group.csv"),
    index=False,
)


# ============================================================
# 16. Save predictions
# ============================================================

predictions_df = X_test.copy()

predictions_df["actual_group"] = actual_group
predictions_df["actual_group_label"] = [group_labels[g] for g in actual_group]

predictions_df["predicted_group"] = pred_group
predictions_df["predicted_group_label"] = [group_labels[g] for g in pred_group]

for i in range(3):
    group = i + 1
    predictions_df[f"prob_group_{group}_{group_labels[group]}"] = pred_prob[:, i]

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_predictions.csv"),
    index=False,
)


# ============================================================
# 17. Save predictor list and group mapping
# ============================================================

pd.DataFrame({"predictor": predictors}).to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_predictors_used.csv"),
    index=False,
)

group_mapping_df = pd.DataFrame(
    [
        {
            "rating_group": 1,
            "rating_group_code": 0,
            "label": "low_or_average_below",
            "original_stars_included": "1.0, 1.5, 2.0, 2.5, 3.0",
            "representative_star_value": 2.50,
        },
        {
            "rating_group": 2,
            "rating_group_code": 1,
            "label": "good_3_5_to_4_0",
            "original_stars_included": "3.5, 4.0",
            "representative_star_value": 3.75,
        },
        {
            "rating_group": 3,
            "rating_group_code": 2,
            "label": "excellent_4_5_to_5_0",
            "original_stars_included": "4.5, 5.0",
            "representative_star_value": 4.75,
        },
    ]
)

group_mapping_df.to_csv(
    os.path.join(OUTPUT_DIR, "ordered_logit_3group_rating_group_mapping.csv"),
    index=False,
)

print("\nAll outputs saved in:")
print(OUTPUT_DIR)