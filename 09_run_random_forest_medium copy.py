import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = "/Users/robertchen/Desktop/UChicago Spring 2026/BUSN 20800/Final Project (Auth)/Data"

INPUT_FILE = os.path.join(
    DATA_DIR,
    "philadelphia_no_review_category_attribute_dummy_features_with_review_count.csv"
)

OUTPUT_DIR = os.path.join(DATA_DIR, "random_forest_outputs_medium")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def round_to_half_star(x):
    return np.clip(np.round(x * 2) / 2, 1.0, 5.0)


df = pd.read_csv(INPUT_FILE)

if "stars" not in df.columns:
    raise ValueError("Dataset must contain 'stars' as the target variable.")

df = df.dropna(subset=["stars"]).copy()


# ------------------------------------------------------------
# Create log count variables
# ------------------------------------------------------------

for col in ["review_count", "checkin_count", "tip_count", "photo_count"]:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])


# ------------------------------------------------------------
# Reconstruct numeric price range
# ------------------------------------------------------------

df["price_range"] = np.nan

price_map = {
    "attr_restaurantspricerange2_1": 1,
    "attr_restaurantspricerange2_2": 2,
    "attr_restaurantspricerange2_3": 3,
    "attr_restaurantspricerange2_4": 4,
}

for col, value in price_map.items():
    if col in df.columns:
        df.loc[df[col] == 1, "price_range"] = value


# ------------------------------------------------------------
# Medium-size random forest predictor set
# ------------------------------------------------------------

candidate_predictors = [
    # Basic business/location variables
    "latitude",
    "longitude",
    "is_open",
    "number_of_categories",
    "price_range",

    # Engagement variables
    "log_review_count",
    "log_checkin_count",
    "log_tip_count",
    "log_photo_count",
    "avg_tip_length",
    "avg_tip_compliments",
    "active_checkin_days",

    # Photo variables
    "photo_file_coverage",
    "avg_image_brightness",
    "avg_image_saturation",
    "food_photo_share",
    "inside_photo_share",
    "outside_photo_share",
    "drink_photo_share",
    "menu_photo_share",

    # Service variables
    "has_takeout",
    "has_delivery",

    # Core attribute variables
    "attr_outdoorseating",
    "attr_bikeparking",
    "attr_restaurantsreservations",
    "attr_restaurantstableservice",
    "attr_goodforkids",
    "attr_restaurantsgoodforgroups",
    "attr_hastv",
    "attr_dogsallowed",
    "attr_wheelchairaccessible",
    "attr_businessacceptscreditcards",
    "attr_caters",
    "attr_happyhour",

    # Alcohol and WiFi variables
    "attr_alcohol_beer_and_wine",
    "attr_alcohol_full_bar",
    "attr_alcohol_none",
    "attr_wifi_free",
    "attr_wifi_no",

    # Common category variables
    "cat_food",
    "cat_nightlife",
    "cat_bars",
    "cat_sandwiches",
    "cat_pizza",
    "cat_american_new",
    "cat_breakfast_brunch",
    "cat_american_traditional",
    "cat_coffee_tea",
    "cat_italian",
    "cat_chinese",
    "cat_fast_food",
    "cat_burgers",
    "cat_seafood",
    "cat_cafes",
    "cat_mexican",
    "cat_delis",
    "cat_salad",
    "cat_chicken_wings",
    "cat_bakeries",
    "cat_japanese",
    "cat_asian_fusion",
    "cat_vegetarian",
    "cat_desserts",
]

predictors = [col for col in candidate_predictors if col in df.columns]
missing_predictors = [col for col in candidate_predictors if col not in df.columns]


# ------------------------------------------------------------
# Prepare modeling data
# ------------------------------------------------------------

model_df = df[["stars"] + predictors].copy()

for col in predictors:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

predictors = [col for col in predictors if not model_df[col].isna().all()]

for col in predictors:
    model_df[col] = model_df[col].fillna(model_df[col].median())

X = model_df[predictors]
y = model_df["stars"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)


# ------------------------------------------------------------
# Train tuned random forest
# ------------------------------------------------------------

param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [8, 10, 14, None],
    "min_samples_leaf": [3, 5, 10],
    "max_features": ["sqrt", 0.5, None],
}

grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_


# ------------------------------------------------------------
# Evaluate model
# ------------------------------------------------------------

pred = best_rf.predict(X_test)
pred_rounded = round_to_half_star(pred)

metrics_df = pd.DataFrame([
    {
        "model": "random_forest_medium_tuned",
        "num_observations": len(model_df),
        "num_train": len(X_train),
        "num_test": len(X_test),
        "num_predictors": len(predictors),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": rmse(y_test, pred),
        "r_squared": r2_score(y_test, pred),
        "rounded_mae": mean_absolute_error(y_test, pred_rounded),
        "rounded_rmse": rmse(y_test, pred_rounded),
        "rounded_r_squared": r2_score(y_test, pred_rounded),
        "within_half_star": np.mean(np.abs(pred_rounded - y_test) <= 0.5),
        "within_one_star": np.mean(np.abs(pred_rounded - y_test) <= 1.0),
    }
])


# ------------------------------------------------------------
# Save tables
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "predictor": predictors,
    "importance": best_rf.feature_importances_,
}).sort_values("importance", ascending=False)

importance_df["rank"] = range(1, len(importance_df) + 1)
importance_df["importance_percent"] = importance_df["importance"] * 100
importance_df = importance_df[["rank", "predictor", "importance", "importance_percent"]]

predictions_df = X_test.copy()
predictions_df["actual_stars"] = y_test.values
predictions_df["predicted_stars"] = pred
predictions_df["predicted_stars_rounded_half_star"] = pred_rounded
predictions_df["prediction_error"] = predictions_df["predicted_stars"] - predictions_df["actual_stars"]
predictions_df["absolute_error"] = predictions_df["prediction_error"].abs()

predictor_df = pd.DataFrame({"predictor": predictors})
missing_predictor_df = pd.DataFrame({"missing_predictor": missing_predictors})
best_params_df = pd.DataFrame([grid.best_params_])

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_medium_metrics.csv"),
    index=False,
)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_medium_feature_importance.csv"),
    index=False,
)

importance_df.head(20).to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_medium_top20_feature_importance.csv"),
    index=False,
)

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_medium_predictions.csv"),
    index=False,
)

predictor_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_medium_predictors_used.csv"),
    index=False,
)

missing_predictor_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_medium_missing_predictors.csv"),
    index=False,
)

best_params_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_medium_best_parameters.csv"),
    index=False,
)


# ------------------------------------------------------------
# Feature importance graph
# ------------------------------------------------------------

top_n = min(20, len(importance_df))
plot_df = importance_df.head(top_n).copy()
plot_df = plot_df.sort_values("importance_percent", ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(plot_df["predictor"], plot_df["importance_percent"])
plt.xlabel("Feature importance (%)")
plt.ylabel("Predictor")
plt.title("Top Random Forest Feature Importances")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_medium_top20_feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# Optional: Actual vs predicted graph
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.4)
plt.xlabel("Actual stars")
plt.ylabel("Predicted stars")
plt.title("Random Forest: Actual vs Predicted Stars")

min_val = min(y_test.min(), pred.min())
max_val = max(y_test.max(), pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_medium_actual_vs_predicted.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print("\nPredictors used:")
for col in predictors:
    print("-", col)

print("\nNumber of predictors used:", len(predictors))

if missing_predictors:
    print("\nCandidate predictors not found and skipped:")
    for col in missing_predictors:
        print("-", col)

print("\nBest parameters:")
print(grid.best_params_)

print("\nSelected max_depth:")
print(grid.best_params_["max_depth"])

print("\nTest metrics:")
print(metrics_df)

print("\nTop feature importances:")
print(importance_df.head(20))

print("\nOutputs saved in:")
print(OUTPUT_DIR)