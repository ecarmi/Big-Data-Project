import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)


DATA_DIR = "/Users/robertchen/Desktop/UChicago Spring 2026/BUSN 20800/Final Project (Auth)/Data"

INPUT_FILE = os.path.join(
    DATA_DIR,
    "philadelphia_no_review_category_attribute_dummy_features_with_review_count.csv"
)

OUTPUT_DIR = os.path.join(DATA_DIR, "random_forest_outputs_90_predictors")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def round_to_half_star(x):
    return np.clip(np.round(x * 2) / 2, 1.0, 5.0)


def excluded(col):
    col = col.lower()
    bad_terms = [
        "business_id", "stars", "rating_group", "sentiment",
        "review_text", "text", "cluster", "zip", "postal",
        "hour", "hours", "open_monday", "open_tuesday",
        "open_wednesday", "open_thursday", "open_friday",
        "open_saturday", "open_sunday", "open_weekend",
        "attr_restaurantspricerange2_",
    ]
    return any(term in col for term in bad_terms)


df = pd.read_csv(INPUT_FILE)

if "stars" not in df.columns:
    raise ValueError("Dataset must contain 'stars'.")

df = df.dropna(subset=["stars"]).copy()


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

for col in ["review_count", "checkin_count", "tip_count", "photo_count"]:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])

df["price_range"] = np.nan

for col, value in {
    "attr_restaurantspricerange2_1": 1,
    "attr_restaurantspricerange2_2": 2,
    "attr_restaurantspricerange2_3": 3,
    "attr_restaurantspricerange2_4": 4,
}.items():
    if col in df.columns:
        df.loc[df[col] == 1, "price_range"] = value


# ------------------------------------------------------------
# Build reasonable candidate predictor pool
# ------------------------------------------------------------

core_predictors = [
    "latitude", "longitude", "is_open", "number_of_categories", "price_range",
    "log_review_count", "log_checkin_count", "log_tip_count", "log_photo_count",
    "avg_tip_length", "avg_tip_compliments", "active_checkin_days",
    "photo_file_coverage", "avg_image_brightness", "avg_image_saturation",
    "food_photo_share", "inside_photo_share", "outside_photo_share",
    "drink_photo_share", "menu_photo_share",
    "has_takeout", "has_delivery",
]

core_predictors = [
    col for col in core_predictors
    if col in df.columns and not excluded(col)
]

dummy_rows = []

for col in df.columns:
    if not (col.startswith("cat_") or col.startswith("attr_")):
        continue

    if excluded(col):
        continue

    x = pd.to_numeric(df[col], errors="coerce")

    if x.isna().all():
        continue

    prevalence = x.mean(skipna=True)

    if pd.isna(prevalence):
        continue

    if 0.02 <= prevalence <= 0.98:
        dummy_rows.append({
            "predictor": col,
            "prevalence": prevalence,
            "distance_from_half": abs(prevalence - 0.5),
        })

dummy_df = pd.DataFrame(dummy_rows)

if len(dummy_df) > 0:
    attr_pool = (
        dummy_df[dummy_df["predictor"].str.startswith("attr_")]
        .sort_values(["distance_from_half", "prevalence"], ascending=[True, False])
        ["predictor"]
        .head(65)
        .tolist()
    )

    cat_pool = (
        dummy_df[dummy_df["predictor"].str.startswith("cat_")]
        .sort_values(["prevalence", "distance_from_half"], ascending=[False, True])
        ["predictor"]
        .head(65)
        .tolist()
    )
else:
    attr_pool = []
    cat_pool = []

candidate_predictors = []

for col in core_predictors + attr_pool + cat_pool:
    if col not in candidate_predictors:
        candidate_predictors.append(col)

candidate_predictors = candidate_predictors[:110]


# ------------------------------------------------------------
# Prepare modeling data
# ------------------------------------------------------------

model_df = df[["stars"] + candidate_predictors].copy()

for col in candidate_predictors:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

predictors = []
imputation_rows = []

for col in candidate_predictors:
    if model_df[col].isna().all():
        continue

    if model_df[col].nunique(dropna=True) <= 1:
        continue

    missing_count = model_df[col].isna().sum()
    median_value = model_df[col].median()

    model_df[col] = model_df[col].fillna(median_value)

    predictors.append(col)
    imputation_rows.append({
        "predictor": col,
        "missing_values_imputed": missing_count,
        "median_used": median_value,
    })

X = model_df[predictors].copy()
y = model_df["stars"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ------------------------------------------------------------
# Select top 90 predictors using training-set RF importance
# ------------------------------------------------------------

if len(predictors) < 90:
    raise ValueError(f"Only {len(predictors)} usable predictors available; need at least 90.")

selector = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

selector.fit(X_train, y_train)

ranking_df = pd.DataFrame({
    "predictor": predictors,
    "selection_importance": selector.feature_importances_,
}).sort_values("selection_importance", ascending=False)

selected_predictors = ranking_df["predictor"].head(90).tolist()


# ------------------------------------------------------------
# Tune final 90-predictor random forest
# ------------------------------------------------------------

param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [10, 14, None],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", 0.5],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train[selected_predictors], y_train)

best_rf = grid.best_estimator_


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

pred = best_rf.predict(X_test[selected_predictors])
pred_rounded = round_to_half_star(pred)

metrics_df = pd.DataFrame([{
    "model": "random_forest_90_predictors",
    "num_candidate_predictors": len(predictors),
    "num_predictors": len(selected_predictors),
    "num_observations": len(model_df),
    "num_train": len(X_train),
    "num_test": len(X_test),
    "cv_r_squared": grid.best_score_,
    "test_r_squared": r2_score(y_test, pred),
    "test_mae": mean_absolute_error(y_test, pred),
    "test_rmse": rmse(y_test, pred),
    "rounded_mae": mean_absolute_error(y_test, pred_rounded),
    "rounded_rmse": rmse(y_test, pred_rounded),
    "rounded_r_squared": r2_score(y_test, pred_rounded),
    "within_half_star": np.mean(np.abs(pred_rounded - y_test) <= 0.5),
    "within_one_star": np.mean(np.abs(pred_rounded - y_test) <= 1.0),
    "best_n_estimators": grid.best_params_["n_estimators"],
    "best_max_depth": grid.best_params_["max_depth"],
    "best_min_samples_leaf": grid.best_params_["min_samples_leaf"],
    "best_max_features": grid.best_params_["max_features"],
}])


importance_df = pd.DataFrame({
    "predictor": selected_predictors,
    "importance": best_rf.feature_importances_,
}).sort_values("importance", ascending=False)

importance_df["rank"] = range(1, len(importance_df) + 1)
importance_df["importance_percent"] = importance_df["importance"] * 100
importance_df = importance_df[["rank", "predictor", "importance", "importance_percent"]]

predictions_df = X_test[selected_predictors].copy()
predictions_df["actual_stars"] = y_test.values
predictions_df["predicted_stars"] = pred
predictions_df["predicted_stars_rounded_half_star"] = pred_rounded
predictions_df["prediction_error"] = predictions_df["predicted_stars"] - predictions_df["actual_stars"]
predictions_df["absolute_error"] = predictions_df["prediction_error"].abs()


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

metrics_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_90_metrics.csv"), index=False)
importance_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_90_feature_importance.csv"), index=False)
importance_df.head(20).to_csv(os.path.join(OUTPUT_DIR, "random_forest_90_top20_feature_importance.csv"), index=False)
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "random_forest_90_predictions.csv"), index=False)
ranking_df.to_csv(os.path.join(OUTPUT_DIR, "predictor_ranking_from_training_rf.csv"), index=False)
pd.DataFrame({"predictor": selected_predictors}).to_csv(os.path.join(OUTPUT_DIR, "random_forest_90_predictors_used.csv"), index=False)
pd.DataFrame(imputation_rows).to_csv(os.path.join(OUTPUT_DIR, "imputation_summary.csv"), index=False)
pd.DataFrame([grid.best_params_]).to_csv(os.path.join(OUTPUT_DIR, "random_forest_90_best_parameters.csv"), index=False)


# ------------------------------------------------------------
# Feature importance graph
# ------------------------------------------------------------

plot_df = importance_df.head(20).sort_values("importance_percent", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(plot_df["predictor"], plot_df["importance_percent"])
plt.xlabel("Feature importance (%)")
plt.ylabel("Predictor")
plt.title("Top 20 Feature Importances: 90-Predictor Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_forest_90_top20_feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Actual vs predicted graph
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.4)
plt.xlabel("Actual stars")
plt.ylabel("Predicted stars")
plt.title("90-Predictor Random Forest: Actual vs Predicted Stars")

min_val = min(y_test.min(), pred.min())
max_val = max(y_test.max(), pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_forest_90_actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()


print("\nBest parameters:")
print(grid.best_params_)

print("\nMetrics:")
print(metrics_df)

print("\nTop 20 feature importances:")
print(importance_df.head(20))

print("\nOutputs saved in:")
print(OUTPUT_DIR)