from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv"
)
OUTPUT_DIR = Path("outputs/ols_lasso_clean_median_imputed_price")
OUTCOME = "stars"
RANDOM_STATE = 42
TEST_SIZE = 0.2


DROP_EXACT = {
    "business_id",
    OUTCOME,
    "avg_tip_compliments",
    "avg_review_length",
    "avg_review_stars",
    "avg_useful",
    "avg_funny",
    "avg_cool",
    "total_review_count",
    "avg_sentiment",
    "avg_review_sentiment_polarity",
    "avg_review_sentiment_subjectivity",
    "positive_review_share",
    "negative_review_share",
    "unique_reviewer_count",
    "avg_reviewer_review_count",
    "avg_reviewer_average_stars",
    "avg_reviewer_fans",
    "avg_reviewer_useful",
    "avg_reviewer_funny",
    "avg_reviewer_cool",
    "avg_reviewer_elite_years",
    "avg_reviewer_friend_count",
    "avg_reviewer_compliments",
    "attr_restaurantstakeout",
    "attr_restaurantsdelivery",
    "photo_file_count",
    "cat_wine_spirits",
    "attr_restaurantspricerange2_1",
    "attr_restaurantspricerange2_2",
    "attr_restaurantspricerange2_3",
    "attr_restaurantspricerange2_nan",
}

REFERENCE_LEVELS = {
    "attr_alcohol_nan",
    "attr_restaurantsattire_nan",
    "attr_wifi_nan",
    "attr_noiselevel_nan",
    "attr_byobcorkage_nan",
    "attr_smoking_nan",
}


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def write_metrics(path: Path, y_train, train_predictions, y_test, test_predictions) -> pd.DataFrame:
    metrics = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(y_train),
                "r_squared": r2_score(y_train, train_predictions),
                "rmse": rmse(y_train, train_predictions),
                "mae": mean_absolute_error(y_train, train_predictions),
            },
            {
                "split": "test",
                "rows": len(y_test),
                "r_squared": r2_score(y_test, test_predictions),
                "rmse": rmse(y_test, test_predictions),
                "mae": mean_absolute_error(y_test, test_predictions),
            },
        ]
    )
    metrics.to_csv(path, index=False)
    return metrics


def drop_rank_dependent_columns(data: pd.DataFrame, predictors: list[str]) -> tuple[list[str], list[str]]:
    x = data[predictors].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(x)
    design = np.column_stack([np.ones(len(data)), scaled])
    _, r_matrix, pivots = qr(design, mode="economic", pivoting=True)
    diagonal = np.diag(r_matrix)
    tolerance = np.finfo(float).eps * max(design.shape) * abs(diagonal).max()
    rank = int((abs(diagonal) > tolerance).sum())
    kept_pivot_positions = set(pivots[:rank])

    kept = []
    dropped = []
    for index, predictor in enumerate(predictors, start=1):
        if index in kept_pivot_positions:
            kept.append(predictor)
        else:
            dropped.append(predictor)
    return kept, dropped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(INPUT_PATH)

    price_median = features["price_range"].median(skipna=True)
    features["price_range_missing"] = features["price_range"].isna().astype(int)
    features["price_range"] = features["price_range"].fillna(price_median)

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    initial_predictors = [
        column
        for column in numeric_columns
        if column not in DROP_EXACT
        and column not in REFERENCE_LEVELS
        and not column.startswith("zip_")
        and "sentiment" not in column.lower()
    ]

    pre_model_data = (
        features[["business_id", OUTCOME] + initial_predictors]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    constant_columns = [
        column for column in initial_predictors if pre_model_data[column].nunique(dropna=False) <= 1
    ]
    predictors_without_constants = [
        column for column in initial_predictors if column not in constant_columns
    ]
    predictors, rank_dropped_columns = drop_rank_dependent_columns(
        pre_model_data, predictors_without_constants
    )

    model_data = pre_model_data[["business_id", OUTCOME] + predictors].copy()
    model_data.to_csv(OUTPUT_DIR / "modeling_dataset_clean.csv", index=False)

    dropped_columns = pd.DataFrame(
        {
            "column": (
                sorted((set(numeric_columns) & DROP_EXACT) - {"business_id", OUTCOME})
                + sorted(set(numeric_columns) & REFERENCE_LEVELS)
                + constant_columns
                + rank_dropped_columns
            ),
            "reason": (
                ["excluded_or_duplicate"] * len(sorted((set(numeric_columns) & DROP_EXACT) - {"business_id", OUTCOME}))
                + ["reference_level"] * len(sorted(set(numeric_columns) & REFERENCE_LEVELS))
                + ["constant"] * len(constant_columns)
                + ["rank_dependent"] * len(rank_dropped_columns)
            ),
        }
    )
    dropped_columns.to_csv(OUTPUT_DIR / "dropped_columns.csv", index=False)

    train_data, test_data = train_test_split(
        model_data, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    x_train = train_data[predictors]
    y_train = train_data[OUTCOME]
    x_test = test_data[predictors]
    y_test = test_data[OUTCOME]

    x_train_ols = sm.add_constant(x_train, has_constant="add")
    x_test_ols = sm.add_constant(x_test, has_constant="add")
    ols_model = sm.OLS(y_train, x_train_ols).fit()
    ols_train_predictions = ols_model.predict(x_train_ols)
    ols_test_predictions = ols_model.predict(x_test_ols)
    ols_metrics = write_metrics(
        OUTPUT_DIR / "ols_metrics.csv",
        y_train,
        ols_train_predictions,
        y_test,
        ols_test_predictions,
    )

    ols_conf_int = ols_model.conf_int()
    pd.DataFrame(
        {
            "term": ols_model.params.index,
            "coefficient": ols_model.params.values,
            "std_error": ols_model.bse.values,
            "t_value": ols_model.tvalues.values,
            "p_value": ols_model.pvalues.values,
            "conf_int_low": ols_conf_int[0].values,
            "conf_int_high": ols_conf_int[1].values,
        }
    ).sort_values("p_value").to_csv(OUTPUT_DIR / "ols_coefficients.csv", index=False)

    pd.DataFrame(
        {
            "business_id": test_data["business_id"].values,
            "actual_stars": y_test.values,
            "predicted_stars": ols_test_predictions.values,
            "residual": y_test.values - ols_test_predictions.values,
        }
    ).to_csv(OUTPUT_DIR / "ols_predictions.csv", index=False)

    lasso_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=10, random_state=RANDOM_STATE, max_iter=100000, alphas=200)),
        ]
    )
    lasso_pipeline.fit(x_train, y_train)
    lasso_train_predictions = lasso_pipeline.predict(x_train)
    lasso_test_predictions = lasso_pipeline.predict(x_test)
    lasso_metrics = write_metrics(
        OUTPUT_DIR / "lasso_metrics.csv",
        y_train,
        lasso_train_predictions,
        y_test,
        lasso_test_predictions,
    )

    scaler = lasso_pipeline.named_steps["scaler"]
    lasso = lasso_pipeline.named_steps["lasso"]
    lasso_coefficients = pd.DataFrame(
        {
            "term": predictors,
            "scaled_coefficient": lasso.coef_,
            "unscaled_coefficient": lasso.coef_ / scaler.scale_,
            "selected": lasso.coef_ != 0,
            "abs_scaled_coefficient": np.abs(lasso.coef_),
        }
    ).sort_values(["selected", "abs_scaled_coefficient"], ascending=[False, False])
    lasso_coefficients.to_csv(OUTPUT_DIR / "lasso_coefficients.csv", index=False)

    pd.DataFrame(
        {
            "business_id": test_data["business_id"].values,
            "actual_stars": y_test.values,
            "predicted_stars": lasso_test_predictions,
            "residual": y_test.values - lasso_test_predictions,
        }
    ).to_csv(OUTPUT_DIR / "lasso_predictions.csv", index=False)

    selected_lasso = lasso_coefficients[lasso_coefficients["selected"]].copy()

    for model_name, metrics, summary_filename in [
        ("OLS", ols_metrics, "ols_summary.txt"),
        ("Lasso", lasso_metrics, "lasso_summary.txt"),
    ]:
        with (OUTPUT_DIR / summary_filename).open("w", encoding="utf-8") as summary_file:
            summary_file.write(
                f"{model_name}: cleaned model with median-imputed price_range, excluding "
                "review length, tip compliment, and ZIP dummies\n"
            )
            summary_file.write(f"Input file: {INPUT_PATH}\n")
            summary_file.write("Included review exception: log_review_count\n")
            summary_file.write(f"Median price_range used for imputation: {price_median:.5f}\n")
            summary_file.write("Added indicator: price_range_missing\n")
            summary_file.write(
                "Cleaned by dropping exact duplicates, constants, one-hot reference "
                "levels, and remaining rank-dependent columns.\n"
            )
            if model_name == "Lasso":
                summary_file.write("Preprocessing: StandardScaler before LassoCV\n")
                summary_file.write("CV folds: 10\n")
                summary_file.write(f"Chosen alpha: {lasso.alpha_:.8f}\n")
                summary_file.write(f"Selected nonzero predictors: {len(selected_lasso):,}\n")
            summary_file.write(f"Rows used: {len(model_data):,}\n")
            summary_file.write(f"Train rows: {len(train_data):,}\n")
            summary_file.write(f"Test rows: {len(test_data):,}\n")
            summary_file.write(f"Initial candidate predictors: {len(initial_predictors):,}\n")
            summary_file.write(f"Final predictors used: {len(predictors):,}\n")
            summary_file.write(f"Dropped columns: {len(dropped_columns):,}\n\n")
            summary_file.write("Metrics:\n")
            summary_file.write(metrics.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
            summary_file.write("\n\nPredictors:\n")
            for predictor in predictors:
                summary_file.write(f"- {predictor}\n")
            if model_name == "OLS":
                summary_file.write("\nModel summary:\n")
                summary_file.write(ols_model.summary().as_text())
            else:
                summary_file.write("\nSelected predictors by absolute scaled coefficient:\n")
                summary_file.write(
                    selected_lasso[["term", "scaled_coefficient", "unscaled_coefficient"]].to_string(
                        index=False, float_format=lambda value: f"{value:.5f}"
                    )
                )
                summary_file.write("\n")

    print("OLS metrics:")
    print(ols_metrics.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nLasso metrics:")
    print(lasso_metrics.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nRows used: {len(model_data):,}")
    print(f"Median price_range: {price_median:.5f}")
    print(f"Final predictors used: {len(predictors):,}")
    print(f"Lasso selected predictors: {len(selected_lasso):,}")
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
