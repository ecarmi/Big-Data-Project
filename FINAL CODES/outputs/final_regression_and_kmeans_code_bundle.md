# Final Regression and K-means Reproducibility Code

This file collects the code used to build the final Philadelphia restaurant analysis pipeline: consolidated data, dummy features, final OLS/Lasso regressions, final 5-variable K-means, K-means graphs, PCA loadings, and K-means analysis notes.

Run order:
1. `src/consolidate_philadelphia_restaurants.py`
2. `src/build_no_review_category_attribute_dummy_features.py`
3. `src/add_log_review_count_to_dummy_features.py`
4. `src/add_price_range_to_kmeans_features.py`
5. `src/run_ols_lasso_clean_median_imputed_price.py`
6. `src/run_kmeans_silhouette_5_variables_k4.py`
7. `src/plot_kmeans_silhouette_5var_k4_2d.py`
8. `src/write_kmeans_5var_analysis_notes.py`

`src/add_engagement_features.py` is included first because the consolidation script imports its check-in/photo helper functions.


## src\add_engagement_features.py

```python
from pathlib import Path

import pandas as pd
from PIL import Image, ImageStat


def find_photo_json(raw_dir: Path) -> Path | None:
    """Return the first available Yelp photo metadata file."""
    candidates = [
        raw_dir / "photos.json",
        raw_dir / "yelp_academic_dataset_photo.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def build_checkin_features(raw_dir: Path, restaurant_ids: set[str]) -> pd.DataFrame:
    checkins_path = raw_dir / "yelp_academic_dataset_checkin.json"
    if not checkins_path.exists():
        return pd.DataFrame(
            columns=["business_id", "checkin_count", "active_checkin_days"]
        )

    checkins = pd.read_json(checkins_path, lines=True)
    checkins = checkins[checkins["business_id"].isin(restaurant_ids)].copy()
    checkins["date_list"] = checkins["date"].fillna("").apply(
        lambda value: [item.strip() for item in value.split(",") if item.strip()]
    )
    checkins["checkin_count"] = checkins["date_list"].apply(len)
    checkins["active_checkin_days"] = checkins["date_list"].apply(
        lambda dates: len({timestamp.split()[0] for timestamp in dates})
    )

    return checkins[["business_id", "checkin_count", "active_checkin_days"]].copy()


def build_photo_features(raw_dir: Path, restaurant_ids: set[str]) -> pd.DataFrame:
    photo_path = find_photo_json(raw_dir)
    feature_columns = [
        "business_id",
        "photo_count",
        "photo_file_count",
        "photo_file_coverage",
        "avg_image_brightness",
        "avg_image_saturation",
        "food_photo_share",
        "inside_photo_share",
        "outside_photo_share",
        "drink_photo_share",
        "menu_photo_share",
    ]
    if photo_path is None:
        return pd.DataFrame(columns=feature_columns)

    photos = pd.read_json(photo_path, lines=True)
    if "label" not in photos.columns:
        label_columns = [col for col in photos.columns if "label" in col.lower()]
        if label_columns:
            photos["label"] = photos[label_columns[0]]
        else:
            raise KeyError("Expected a label column in the photo JSON file")

    photos["label"] = photos["label"].astype(str).str.lower().str.strip()
    photos = photos[photos["business_id"].isin(restaurant_ids)].copy()

    photo_dir = raw_dir / "photos"
    if photo_dir.exists() and "photo_id" in photos.columns:
        image_cache_path = raw_dir.parent / "processed" / "image_color_features.csv"
        needed_photo_ids = set(photos["photo_id"].dropna())
        image_features = build_image_features(photo_dir, image_cache_path, needed_photo_ids)
        available_photo_ids = set(image_features["photo_id"])
        photos["has_photo_file"] = photos["photo_id"].isin(available_photo_ids).astype(int)
        photos = photos.merge(image_features, on="photo_id", how="left")
        photo_file_count = photos.groupby("business_id")["has_photo_file"].sum()
        image_feature_means = photos.groupby("business_id").agg(
            avg_image_brightness=("image_brightness", "mean"),
            avg_image_saturation=("image_saturation", "mean"),
        )
    else:
        photo_file_count = pd.Series(dtype="int64")
        image_feature_means = pd.DataFrame(
            columns=["avg_image_brightness", "avg_image_saturation"]
        )

    photo_count = photos.groupby("business_id").size().rename("photo_count")
    label_counts = photos.groupby(["business_id", "label"]).size().unstack(fill_value=0)

    photo_features = label_counts.copy()
    photo_features["photo_count"] = photo_count
    photo_features["photo_file_count"] = photo_file_count
    photo_features["photo_file_count"] = photo_features["photo_file_count"].fillna(0)
    photo_features["photo_file_coverage"] = (
        photo_features["photo_file_count"] / photo_features["photo_count"]
    )
    photo_features = photo_features.join(image_feature_means, how="left")
    for label in ["food", "inside", "outside", "drink", "menu"]:
        photo_features[f"{label}_photo_share"] = (
            photo_features.get(label, 0) / photo_features["photo_count"]
        )

    return photo_features.reset_index()[feature_columns]


def compute_image_color_features(image_path: Path) -> dict[str, float | str]:
    """Compute average brightness and saturation for one image.

    Brightness is the mean HSV value channel and saturation is the mean HSV
    saturation channel. Both are scaled from 0 to 1.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((128, 128))
        hsv_image = image.convert("HSV")
        saturation, brightness = ImageStat.Stat(hsv_image).mean[1:3]

    return {
        "photo_id": image_path.stem,
        "image_brightness": brightness / 255,
        "image_saturation": saturation / 255,
    }


def build_image_features(
    photo_dir: Path,
    cache_path: Path,
    needed_photo_ids: set[str],
) -> pd.DataFrame:
    if cache_path.exists():
        cached_features = pd.read_csv(cache_path)
    else:
        cached_features = pd.DataFrame(
            columns=["photo_id", "image_brightness", "image_saturation"]
        )

    cached_photo_ids = set(cached_features["photo_id"].dropna())
    missing_photo_ids = needed_photo_ids - cached_photo_ids
    image_paths = [
        image_path
        for image_path in photo_dir.glob("*.jpg")
        if image_path.stem in missing_photo_ids
    ]

    rows = []
    for image_number, image_path in enumerate(image_paths, start=1):
        try:
            rows.append(compute_image_color_features(image_path))
        except OSError as error:
            print(f"Skipping unreadable image {image_path.name}: {error}")

        if image_number % 25000 == 0:
            print(f"Processed {image_number:,} new image files")

    if rows:
        new_features = pd.DataFrame(rows)
        image_features = pd.concat([cached_features, new_features], ignore_index=True)
        image_features = image_features.drop_duplicates("photo_id", keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        image_features.to_csv(cache_path, index=False)
    else:
        image_features = cached_features

    return image_features[image_features["photo_id"].isin(needed_photo_ids)].copy()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    input_path = processed_dir / "restaurant_review_features.csv"
    output_path = processed_dir / "restaurant_review_checkin_photo_features.csv"

    restaurants = pd.read_csv(input_path)
    restaurant_ids = set(restaurants["business_id"].dropna())

    checkin_features = build_checkin_features(raw_dir, restaurant_ids)
    photo_features = build_photo_features(raw_dir, restaurant_ids)

    enriched = restaurants.merge(checkin_features, on="business_id", how="left")
    enriched = enriched.merge(photo_features, on="business_id", how="left")

    fill_zero_columns = [
        "checkin_count",
        "active_checkin_days",
        "photo_count",
        "photo_file_count",
        "photo_file_coverage",
        "avg_image_brightness",
        "avg_image_saturation",
        "food_photo_share",
        "inside_photo_share",
        "outside_photo_share",
        "drink_photo_share",
        "menu_photo_share",
    ]
    enriched[fill_zero_columns] = enriched[fill_zero_columns].fillna(0)

    enriched.to_csv(output_path, index=False)
    print(f"Wrote {len(enriched):,} rows to {output_path}")


if __name__ == "__main__":
    main()
```

## src\consolidate_philadelphia_restaurants.py

```python
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from textblob import TextBlob

from add_engagement_features import build_checkin_features, build_photo_features


TARGET_CITY = "Philadelphia"
CHUNK_SIZE = 100_000


def count_words(text: object) -> int:
    return len(str(text).split())


def sentiment_polarity(text: object) -> float:
    return TextBlob(str(text)).sentiment.polarity


def sentiment_subjectivity(text: object) -> float:
    return TextBlob(str(text)).sentiment.subjectivity


def attribute_is_true(attributes: object, key: str) -> int:
    if not isinstance(attributes, dict):
        return 0
    return int(str(attributes.get(key)).casefold() == "true")


def split_count(value: object) -> int:
    if pd.isna(value):
        return 0
    return len([item.strip() for item in str(value).split(",") if item.strip()])


def load_philadelphia_restaurants(raw_dir: Path) -> pd.DataFrame:
    business_path = raw_dir / "yelp_academic_dataset_business.json"
    business = pd.read_json(business_path, lines=True)
    categories = business["categories"].fillna("")
    restaurants = business[
        categories.str.contains("Restaurants", case=False, na=False)
    ].copy()
    restaurants = restaurants[
        restaurants["city"].fillna("").str.casefold() == TARGET_CITY.casefold()
    ].copy()

    restaurants["review_count"] = pd.to_numeric(
        restaurants["review_count"], errors="coerce"
    ).fillna(0)
    restaurants["log_review_count"] = restaurants["review_count"].apply(math.log1p)
    restaurants["has_takeout"] = restaurants["attributes"].apply(
        lambda attrs: attribute_is_true(attrs, "RestaurantsTakeOut")
    )
    restaurants["has_delivery"] = restaurants["attributes"].apply(
        lambda attrs: attribute_is_true(attrs, "RestaurantsDelivery")
    )
    restaurants["number_of_categories"] = restaurants["categories"].apply(split_count)
    return restaurants


def build_review_features(
    raw_dir: Path, restaurant_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_path = raw_dir / "yelp_academic_dataset_review.json"
    chunk_summaries = []
    business_user_pairs = []

    for chunk_number, review_chunk in enumerate(
        pd.read_json(review_path, lines=True, chunksize=CHUNK_SIZE), start=1
    ):
        reviews = review_chunk[review_chunk["business_id"].isin(restaurant_ids)].copy()
        if reviews.empty:
            continue

        for column in ["stars", "useful", "funny", "cool"]:
            reviews[column] = pd.to_numeric(reviews[column], errors="coerce").fillna(0)

        reviews["review_word_count"] = reviews["text"].apply(count_words)
        reviews["review_sentiment_polarity"] = reviews["text"].apply(sentiment_polarity)
        reviews["review_sentiment_subjectivity"] = reviews["text"].apply(
            sentiment_subjectivity
        )
        reviews["positive_review"] = (reviews["review_sentiment_polarity"] > 0).astype(
            int
        )
        reviews["negative_review"] = (reviews["review_sentiment_polarity"] < 0).astype(
            int
        )

        chunk_summaries.append(
            reviews.groupby("business_id", as_index=False).agg(
                review_stars_sum=("stars", "sum"),
                review_word_count_sum=("review_word_count", "sum"),
                useful_sum=("useful", "sum"),
                funny_sum=("funny", "sum"),
                cool_sum=("cool", "sum"),
                review_sentiment_polarity_sum=("review_sentiment_polarity", "sum"),
                review_sentiment_subjectivity_sum=(
                    "review_sentiment_subjectivity",
                    "sum",
                ),
                positive_review_sum=("positive_review", "sum"),
                negative_review_sum=("negative_review", "sum"),
                total_review_count=("review_id", "count"),
                first_review_date=("date", "min"),
                last_review_date=("date", "max"),
            )
        )
        business_user_pairs.append(reviews[["business_id", "user_id"]].dropna())

        if chunk_number % 10 == 0:
            print(f"Processed {chunk_number:,} review chunks")

    if not chunk_summaries:
        empty_columns = [
            "business_id",
            "avg_review_stars",
            "avg_review_length",
            "avg_useful",
            "avg_funny",
            "avg_cool",
            "total_review_count",
            "avg_review_sentiment_polarity",
            "avg_review_sentiment_subjectivity",
            "positive_review_share",
            "negative_review_share",
            "first_review_date",
            "last_review_date",
        ]
        return pd.DataFrame(columns=empty_columns), pd.DataFrame(
            columns=["business_id", "user_id"]
        )

    review_sums = pd.concat(chunk_summaries, ignore_index=True)
    review_features = review_sums.groupby("business_id", as_index=False).agg(
        review_stars_sum=("review_stars_sum", "sum"),
        review_word_count_sum=("review_word_count_sum", "sum"),
        useful_sum=("useful_sum", "sum"),
        funny_sum=("funny_sum", "sum"),
        cool_sum=("cool_sum", "sum"),
        review_sentiment_polarity_sum=("review_sentiment_polarity_sum", "sum"),
        review_sentiment_subjectivity_sum=("review_sentiment_subjectivity_sum", "sum"),
        positive_review_sum=("positive_review_sum", "sum"),
        negative_review_sum=("negative_review_sum", "sum"),
        total_review_count=("total_review_count", "sum"),
        first_review_date=("first_review_date", "min"),
        last_review_date=("last_review_date", "max"),
    )
    denominator = review_features["total_review_count"].replace(0, pd.NA)
    review_features["avg_review_stars"] = (
        review_features["review_stars_sum"] / denominator
    )
    review_features["avg_review_length"] = (
        review_features["review_word_count_sum"] / denominator
    )
    review_features["avg_useful"] = review_features["useful_sum"] / denominator
    review_features["avg_funny"] = review_features["funny_sum"] / denominator
    review_features["avg_cool"] = review_features["cool_sum"] / denominator
    review_features["avg_review_sentiment_polarity"] = (
        review_features["review_sentiment_polarity_sum"] / denominator
    )
    review_features["avg_review_sentiment_subjectivity"] = (
        review_features["review_sentiment_subjectivity_sum"] / denominator
    )
    review_features["positive_review_share"] = (
        review_features["positive_review_sum"] / denominator
    )
    review_features["negative_review_share"] = (
        review_features["negative_review_sum"] / denominator
    )
    review_features["avg_sentiment"] = review_features[
        "avg_review_sentiment_polarity"
    ]

    feature_columns = [
        "business_id",
        "avg_review_stars",
        "avg_review_length",
        "avg_useful",
        "avg_funny",
        "avg_cool",
        "total_review_count",
        "avg_sentiment",
        "avg_review_sentiment_polarity",
        "avg_review_sentiment_subjectivity",
        "positive_review_share",
        "negative_review_share",
        "first_review_date",
        "last_review_date",
    ]
    pairs = pd.concat(business_user_pairs, ignore_index=True).drop_duplicates()
    return review_features[feature_columns], pairs


def build_tip_features(raw_dir: Path, restaurant_ids: set[str]) -> pd.DataFrame:
    tip_path = raw_dir / "yelp_academic_dataset_tip.json"
    chunk_summaries = []

    for chunk_number, tip_chunk in enumerate(
        pd.read_json(tip_path, lines=True, chunksize=CHUNK_SIZE), start=1
    ):
        tips = tip_chunk[tip_chunk["business_id"].isin(restaurant_ids)].copy()
        if tips.empty:
            continue

        tips["compliment_count"] = pd.to_numeric(
            tips["compliment_count"], errors="coerce"
        ).fillna(0)
        tips["tip_word_count"] = tips["text"].apply(count_words)
        tips["tip_sentiment_polarity"] = tips["text"].apply(sentiment_polarity)
        tips["tip_sentiment_subjectivity"] = tips["text"].apply(sentiment_subjectivity)

        chunk_summaries.append(
            tips.groupby("business_id", as_index=False).agg(
                tip_count=("text", "count"),
                tip_word_count_sum=("tip_word_count", "sum"),
                tip_compliment_sum=("compliment_count", "sum"),
                tip_sentiment_polarity_sum=("tip_sentiment_polarity", "sum"),
                tip_sentiment_subjectivity_sum=("tip_sentiment_subjectivity", "sum"),
                first_tip_date=("date", "min"),
                last_tip_date=("date", "max"),
            )
        )

        if chunk_number % 10 == 0:
            print(f"Processed {chunk_number:,} tip chunks")

    columns = [
        "business_id",
        "tip_count",
        "avg_tip_length",
        "avg_tip_compliments",
        "avg_tip_sentiment_polarity",
        "avg_tip_sentiment_subjectivity",
        "first_tip_date",
        "last_tip_date",
    ]
    if not chunk_summaries:
        return pd.DataFrame(columns=columns)

    tip_sums = pd.concat(chunk_summaries, ignore_index=True)
    tip_features = tip_sums.groupby("business_id", as_index=False).agg(
        tip_count=("tip_count", "sum"),
        tip_word_count_sum=("tip_word_count_sum", "sum"),
        tip_compliment_sum=("tip_compliment_sum", "sum"),
        tip_sentiment_polarity_sum=("tip_sentiment_polarity_sum", "sum"),
        tip_sentiment_subjectivity_sum=("tip_sentiment_subjectivity_sum", "sum"),
        first_tip_date=("first_tip_date", "min"),
        last_tip_date=("last_tip_date", "max"),
    )
    denominator = tip_features["tip_count"].replace(0, pd.NA)
    tip_features["avg_tip_length"] = tip_features["tip_word_count_sum"] / denominator
    tip_features["avg_tip_compliments"] = tip_features["tip_compliment_sum"] / denominator
    tip_features["avg_tip_sentiment_polarity"] = (
        tip_features["tip_sentiment_polarity_sum"] / denominator
    )
    tip_features["avg_tip_sentiment_subjectivity"] = (
        tip_features["tip_sentiment_subjectivity_sum"] / denominator
    )
    return tip_features[columns]


def build_reviewer_user_features(raw_dir: Path, business_user_pairs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "business_id",
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
    ]
    if business_user_pairs.empty:
        return pd.DataFrame(columns=columns)

    user_ids = set(business_user_pairs["user_id"].dropna())
    user_path = raw_dir / "yelp_academic_dataset_user.json"
    needed_user_chunks = []

    for chunk_number, user_chunk in enumerate(
        pd.read_json(user_path, lines=True, chunksize=CHUNK_SIZE), start=1
    ):
        users = user_chunk[user_chunk["user_id"].isin(user_ids)].copy()
        if users.empty:
            continue

        numeric_columns = [
            "review_count",
            "useful",
            "funny",
            "cool",
            "fans",
            "average_stars",
            "compliment_hot",
            "compliment_more",
            "compliment_profile",
            "compliment_cute",
            "compliment_list",
            "compliment_note",
            "compliment_plain",
            "compliment_cool",
            "compliment_funny",
            "compliment_writer",
            "compliment_photos",
        ]
        for column in numeric_columns:
            users[column] = pd.to_numeric(users[column], errors="coerce").fillna(0)

        users["elite_years"] = users["elite"].apply(split_count)
        users["friend_count"] = users["friends"].apply(split_count)
        compliment_columns = [column for column in numeric_columns if column.startswith("compliment_")]
        users["total_compliments"] = users[compliment_columns].sum(axis=1)

        needed_user_chunks.append(
            users[
                [
                    "user_id",
                    "review_count",
                    "average_stars",
                    "fans",
                    "useful",
                    "funny",
                    "cool",
                    "elite_years",
                    "friend_count",
                    "total_compliments",
                ]
            ]
        )

        if chunk_number % 10 == 0:
            print(f"Processed {chunk_number:,} user chunks")

    if not needed_user_chunks:
        return pd.DataFrame(columns=columns)

    users = pd.concat(needed_user_chunks, ignore_index=True).drop_duplicates("user_id")
    reviewer_data = business_user_pairs.merge(users, on="user_id", how="left")
    reviewer_features = reviewer_data.groupby("business_id", as_index=False).agg(
        unique_reviewer_count=("user_id", "nunique"),
        avg_reviewer_review_count=("review_count", "mean"),
        avg_reviewer_average_stars=("average_stars", "mean"),
        avg_reviewer_fans=("fans", "mean"),
        avg_reviewer_useful=("useful", "mean"),
        avg_reviewer_funny=("funny", "mean"),
        avg_reviewer_cool=("cool", "mean"),
        avg_reviewer_elite_years=("elite_years", "mean"),
        avg_reviewer_friend_count=("friend_count", "mean"),
        avg_reviewer_compliments=("total_compliments", "mean"),
    )
    return reviewer_features[columns]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    output_path = processed_dir / "philadelphia_restaurants_consolidated.csv"

    processed_dir.mkdir(parents=True, exist_ok=True)

    restaurants = load_philadelphia_restaurants(raw_dir)
    restaurant_ids = set(restaurants["business_id"].dropna())
    print(f"Loaded {len(restaurants):,} Philadelphia restaurants")

    review_features, business_user_pairs = build_review_features(raw_dir, restaurant_ids)
    tip_features = build_tip_features(raw_dir, restaurant_ids)
    checkin_features = build_checkin_features(raw_dir, restaurant_ids)
    photo_features = build_photo_features(raw_dir, restaurant_ids)
    reviewer_features = build_reviewer_user_features(raw_dir, business_user_pairs)

    consolidated = restaurants.merge(review_features, on="business_id", how="left")
    consolidated = consolidated.merge(tip_features, on="business_id", how="left")
    consolidated = consolidated.merge(checkin_features, on="business_id", how="left")
    consolidated = consolidated.merge(photo_features, on="business_id", how="left")
    consolidated = consolidated.merge(reviewer_features, on="business_id", how="left")

    zero_fill_columns = [
        "avg_review_stars",
        "avg_review_length",
        "avg_useful",
        "avg_funny",
        "avg_cool",
        "total_review_count",
        "avg_sentiment",
        "avg_review_sentiment_polarity",
        "avg_review_sentiment_subjectivity",
        "positive_review_share",
        "negative_review_share",
        "tip_count",
        "avg_tip_length",
        "avg_tip_compliments",
        "avg_tip_sentiment_polarity",
        "avg_tip_sentiment_subjectivity",
        "checkin_count",
        "active_checkin_days",
        "photo_count",
        "photo_file_count",
        "photo_file_coverage",
        "avg_image_brightness",
        "avg_image_saturation",
        "food_photo_share",
        "inside_photo_share",
        "outside_photo_share",
        "drink_photo_share",
        "menu_photo_share",
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
    ]
    existing_zero_fill_columns = [
        column for column in zero_fill_columns if column in consolidated.columns
    ]
    consolidated[existing_zero_fill_columns] = consolidated[
        existing_zero_fill_columns
    ].fillna(0)

    consolidated.to_csv(output_path, index=False)
    print(f"Wrote {len(consolidated):,} rows and {len(consolidated.columns):,} columns to {output_path}")


if __name__ == "__main__":
    main()
```

## src\build_no_review_category_attribute_dummy_features.py

```python
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/philadelphia_restaurants_consolidated.csv")
OUTPUT_PATH = Path("outputs/philadelphia_no_review_category_attribute_dummy_features.csv")
CATEGORY_MIN_COUNT = 75
LEVEL_MIN_COUNT = 75


BASE_COLUMNS = [
    "business_id",
    "stars",
    "latitude",
    "longitude",
    "is_open",
    "has_takeout",
    "has_delivery",
    "number_of_categories",
    "tip_count",
    "avg_tip_length",
    "avg_tip_compliments",
    "checkin_count",
    "active_checkin_days",
    "photo_count",
    "photo_file_count",
    "photo_file_coverage",
    "avg_image_brightness",
    "avg_image_saturation",
    "food_photo_share",
    "inside_photo_share",
    "outside_photo_share",
    "drink_photo_share",
    "menu_photo_share",
]


def safe_name(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_").lower()


def parse_literal(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        cleaned = str(value).replace("u'", "'").replace('u"', '"')
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return None


def normalize_scalar(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in {"none", "nan", ""}:
        return None
    if len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0]:
        text = text[1:-1]
    if text.startswith("u'") and text.endswith("'"):
        text = text[2:-1]
    if text.startswith('u"') and text.endswith('"'):
        text = text[2:-1]
    text_lower = text.casefold()
    if text_lower == "true":
        return "true"
    if text_lower == "false":
        return "false"
    return safe_name(text_lower)


def truthy(value: object) -> int:
    return int(normalize_scalar(value) == "true")


def build_category_features(restaurants: pd.DataFrame) -> pd.DataFrame:
    category_lists = restaurants["categories"].fillna("").apply(
        lambda value: [item.strip() for item in str(value).split(",") if item.strip()]
    )
    category_counts = category_lists.explode().value_counts()
    kept_categories = [
        category
        for category, count in category_counts.items()
        if category != "Restaurants" and count >= CATEGORY_MIN_COUNT
    ]

    category_features = pd.DataFrame(index=restaurants.index)
    for category in kept_categories:
        category_features[f"cat_{safe_name(category)}"] = category_lists.apply(
            lambda values, selected=category: int(selected in values)
        )
    return category_features


def build_attribute_features(restaurants: pd.DataFrame) -> pd.DataFrame:
    attrs = restaurants["attributes"].apply(parse_literal)
    attribute_features = pd.DataFrame(index=restaurants.index)

    boolean_attrs = [
        "RestaurantsTakeOut",
        "RestaurantsDelivery",
        "BusinessAcceptsCreditCards",
        "OutdoorSeating",
        "RestaurantsReservations",
        "RestaurantsGoodForGroups",
        "HasTV",
        "GoodForKids",
        "BikeParking",
        "Caters",
        "RestaurantsTableService",
        "HappyHour",
        "WheelchairAccessible",
        "DogsAllowed",
        "BusinessAcceptsBitcoin",
        "Corkage",
        "BYOB",
        "GoodForDancing",
        "CoatCheck",
    ]
    for key in boolean_attrs:
        values = attrs.apply(lambda value, attr_key=key: value.get(attr_key) if isinstance(value, dict) else None)
        if values.notna().sum() >= LEVEL_MIN_COUNT:
            attribute_features[f"attr_{safe_name(key)}"] = values.apply(truthy)
            attribute_features[f"attr_{safe_name(key)}_missing"] = values.isna().astype(int)

    nominal_attrs = [
        "RestaurantsPriceRange2",
        "Alcohol",
        "RestaurantsAttire",
        "WiFi",
        "NoiseLevel",
        "BYOBCorkage",
        "Smoking",
    ]
    for key in nominal_attrs:
        values = attrs.apply(lambda value, attr_key=key: value.get(attr_key) if isinstance(value, dict) else None)
        values = values.apply(normalize_scalar)
        if values.notna().sum() < LEVEL_MIN_COUNT:
            continue
        dummies = pd.get_dummies(values, prefix=f"attr_{safe_name(key)}", dummy_na=True, dtype=int)
        dummies = dummies.loc[:, dummies.sum(axis=0) >= LEVEL_MIN_COUNT]
        attribute_features = pd.concat([attribute_features, dummies], axis=1)

    nested_attrs = {
        "BusinessParking": ["garage", "street", "validated", "lot", "valet"],
        "Ambience": ["touristy", "hipster", "romantic", "divey", "intimate", "trendy", "upscale", "classy", "casual"],
        "GoodForMeal": ["dessert", "latenight", "lunch", "dinner", "brunch", "breakfast"],
        "Music": ["dj", "background_music", "no_music", "jukebox", "live", "video", "karaoke"],
        "BestNights": ["monday", "tuesday", "friday", "wednesday", "thursday", "sunday", "saturday"],
    }
    for parent_key, child_keys in nested_attrs.items():
        parent_values = attrs.apply(
            lambda value, attr_key=parent_key: parse_literal(value.get(attr_key)) if isinstance(value, dict) else None
        )
        for child_key in child_keys:
            values = parent_values.apply(lambda value, key=child_key: value.get(key) if isinstance(value, dict) else None)
            if values.notna().sum() >= LEVEL_MIN_COUNT:
                attribute_features[f"attr_{safe_name(parent_key)}_{child_key}"] = values.apply(truthy)

    return attribute_features


def build_hours_features(restaurants: pd.DataFrame) -> pd.DataFrame:
    hours = restaurants["hours"].apply(parse_literal)
    hours_features = pd.DataFrame(index=restaurants.index)
    hours_features["has_hours"] = hours.apply(lambda value: int(isinstance(value, dict) and len(value) > 0))
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        hours_features[f"open_{day.lower()}"] = hours.apply(
            lambda value, selected_day=day: int(isinstance(value, dict) and selected_day in value)
        )
    hours_features["open_weekend"] = (
        (hours_features["open_saturday"] == 1) | (hours_features["open_sunday"] == 1)
    ).astype(int)
    return hours_features


def build_postal_features(restaurants: pd.DataFrame) -> pd.DataFrame:
    postal = restaurants["postal_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    postal_counts = postal.value_counts()
    kept_postal = postal_counts[postal_counts >= LEVEL_MIN_COUNT].index
    postal_dummies = pd.get_dummies(
        postal.where(postal.isin(kept_postal), "other"), prefix="zip", dtype=int
    )
    return postal_dummies.drop(columns=["zip_other"], errors="ignore")


def main() -> None:
    restaurants = pd.read_csv(INPUT_PATH)
    missing_base_columns = [column for column in BASE_COLUMNS if column not in restaurants.columns]
    if missing_base_columns:
        raise KeyError(f"Missing required base columns: {missing_base_columns}")

    category_features = build_category_features(restaurants)
    attribute_features = build_attribute_features(restaurants)
    hours_features = build_hours_features(restaurants)
    postal_features = build_postal_features(restaurants)

    features = pd.concat(
        [
            restaurants[BASE_COLUMNS],
            category_features,
            attribute_features,
            hours_features,
            postal_features,
        ],
        axis=1,
    )
    features = features.loc[:, ~features.columns.duplicated()].copy()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(features):,} rows and {len(features.columns):,} columns to {OUTPUT_PATH}")
    print(f"Category dummy columns: {category_features.shape[1]:,}")
    print(f"Attribute dummy columns: {attribute_features.shape[1]:,}")
    print(f"Hours dummy columns: {hours_features.shape[1]:,}")
    print(f"ZIP dummy columns: {postal_features.shape[1]:,}")


if __name__ == "__main__":
    main()
```

## src\add_log_review_count_to_dummy_features.py

```python
from pathlib import Path

import pandas as pd


DUMMY_FEATURES_PATH = Path("outputs/philadelphia_no_review_category_attribute_dummy_features.csv")
CONSOLIDATED_PATH = Path("data/processed/philadelphia_restaurants_consolidated.csv")
OUTPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count.csv"
)


def main() -> None:
    dummy_features = pd.read_csv(DUMMY_FEATURES_PATH)
    log_review_counts = pd.read_csv(
        CONSOLIDATED_PATH, usecols=["business_id", "log_review_count"]
    )

    if "log_review_count" in dummy_features.columns:
        dummy_features = dummy_features.drop(columns=["log_review_count"])

    enriched = dummy_features.merge(log_review_counts, on="business_id", how="left")
    if enriched["log_review_count"].isna().any():
        missing_count = enriched["log_review_count"].isna().sum()
        raise ValueError(f"Missing log_review_count for {missing_count:,} rows")

    # Keep the clustering variable near the original business-level predictors.
    columns = enriched.columns.tolist()
    columns.remove("log_review_count")
    insert_after = columns.index("is_open") + 1
    columns.insert(insert_after, "log_review_count")
    enriched = enriched[columns]

    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(enriched):,} rows and {len(enriched.columns):,} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

## src\add_price_range_to_kmeans_features.py

```python
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count.csv"
)
OUTPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv"
)
PRICE_DUMMY_COLUMNS = {
    "attr_restaurantspricerange2_1": 1,
    "attr_restaurantspricerange2_2": 2,
    "attr_restaurantspricerange2_3": 3,
    "attr_restaurantspricerange2_4": 4,
}


def main() -> None:
    features = pd.read_csv(INPUT_PATH)

    available_price_columns = [
        column for column in PRICE_DUMMY_COLUMNS if column in features.columns
    ]
    if not available_price_columns:
        raise KeyError("No attr_restaurantspricerange2_* dummy columns found")

    features["price_range"] = pd.NA
    for column in available_price_columns:
        features.loc[features[column] == 1, "price_range"] = PRICE_DUMMY_COLUMNS[column]

    columns = features.columns.tolist()
    columns.remove("price_range")
    insert_after = columns.index("log_review_count") + 1
    columns.insert(insert_after, "price_range")
    features = features[columns]

    features.to_csv(OUTPUT_PATH, index=False)
    non_missing = features["price_range"].notna().sum()
    print(f"Wrote {len(features):,} rows and {len(features.columns):,} columns to {OUTPUT_PATH}")
    print(f"Rows with non-missing price_range: {non_missing:,}")
    print("price_range counts:")
    print(features["price_range"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
```

## src\run_ols_lasso_clean_median_imputed_price.py

```python
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
```

## src\run_kmeans_silhouette_5_variables_k4.py

```python
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv"
)
OUTPUT_DIR = Path("outputs/kmeans_5_variable_k4")
OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_clustered.csv"
SUMMARY_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_summary.txt"
FEATURES = [
    "log_review_count",
    "avg_tip_compliments",
    "checkin_count",
    "photo_count",
    "inside_photo_share",
]
N_CLUSTERS = 4
RANDOM_STATE = 42


def star_group(stars: pd.Series) -> pd.Series:
    return pd.cut(
        stars,
        bins=[0, 3.0, 3.5, 5.0],
        labels=["low", "middle", "high"],
        include_lowest=True,
    )


def cluster_majority_accuracy(
    true_groups: pd.Series, cluster_labels
) -> tuple[float, dict[int, str]]:
    cluster_to_group = {}
    predicted_groups = []

    for cluster in sorted(set(cluster_labels)):
        in_cluster = cluster_labels == cluster
        majority_group = true_groups[in_cluster].mode().iloc[0]
        cluster_to_group[int(cluster)] = str(majority_group)

    for cluster in cluster_labels:
        predicted_groups.append(cluster_to_group[int(cluster)])

    return accuracy_score(true_groups.astype(str), predicted_groups), cluster_to_group


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    restaurants = pd.read_csv(INPUT_PATH)
    model_data = restaurants[["business_id", "stars"] + FEATURES].dropna().copy()
    true_groups = star_group(model_data["stars"])

    scaled_features = StandardScaler().fit_transform(model_data[FEATURES])
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=25)
    labels = kmeans.fit_predict(scaled_features)
    model_data["kmeans_cluster"] = labels

    accuracy, cluster_mapping = cluster_majority_accuracy(true_groups, labels)
    silhouette = silhouette_score(scaled_features, labels)

    clustered = restaurants.merge(
        model_data[["business_id", "kmeans_cluster"]], on="business_id", how="left"
    )
    clustered.to_csv(OUTPUT_PATH, index=False)

    cluster_summary = (
        model_data.assign(star_group=true_groups)
        .groupby("kmeans_cluster", as_index=False)
        .agg(
            restaurant_count=("business_id", "count"),
            avg_stars=("stars", "mean"),
            avg_log_review_count=("log_review_count", "mean"),
            avg_tip_compliments=("avg_tip_compliments", "mean"),
            avg_checkin_count=("checkin_count", "mean"),
            avg_photo_count=("photo_count", "mean"),
            avg_inside_photo_share=("inside_photo_share", "mean"),
            low_share=("star_group", lambda values: (values == "low").mean()),
            middle_share=("star_group", lambda values: (values == "middle").mean()),
            high_share=("star_group", lambda values: (values == "high").mean()),
        )
        .sort_values("kmeans_cluster")
    )

    with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        summary_file.write("K-means with 5 variables and k=4\n")
        summary_file.write(f"Input file: {INPUT_PATH}\n")
        summary_file.write(f"Output file: {OUTPUT_PATH}\n")
        summary_file.write(f"Features: {', '.join(FEATURES)}\n")
        summary_file.write("Dropped photo_file_count because it duplicates photo_count in this dataset\n")
        summary_file.write(f"Clusters: {N_CLUSTERS}\n")
        summary_file.write(f"Rows used: {len(model_data):,}\n")
        summary_file.write(f"Rows skipped for missing features: {len(restaurants) - len(model_data):,}\n")
        summary_file.write(f"Silhouette score: {silhouette:.5f}\n")
        summary_file.write(f"Star-group majority accuracy: {accuracy:.5f}\n")
        summary_file.write(f"Cluster to star-group mapping: {cluster_mapping}\n\n")
        summary_file.write("Cluster summary:\n")
        summary_file.write(
            cluster_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}")
        )
        summary_file.write("\n")

    print(f"Silhouette score: {silhouette:.5f}")
    print(f"Star-group majority accuracy: {accuracy:.5f}")
    print(cluster_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Wrote clustered data to {OUTPUT_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
```

## src\plot_kmeans_silhouette_5var_k4_2d.py

```python
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs/kmeans_5_variable_k4")
INPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_clustered.csv"
OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca.png"
TRANSPARENT_OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_transparent.png"
NO_OUTLIERS_OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_no_outliers.png"
NO_OUTLIERS_TRANSPARENT_OUTPUT_PATH = (
    OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_no_outliers_transparent.png"
)
FACET_OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_facets.png"
COORDINATES_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_coordinates.csv"
LOADINGS_CSV_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_pca_loadings.csv"
LOADINGS_TXT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_pca_loadings.txt"
FEATURES = [
    "log_review_count",
    "avg_tip_compliments",
    "checkin_count",
    "photo_count",
    "inside_photo_share",
]


def main() -> None:
    clustered = pd.read_csv(INPUT_PATH)
    plot_data = clustered[["business_id", "stars", "kmeans_cluster"] + FEATURES].dropna().copy()

    scaled_features = StandardScaler().fit_transform(plot_data[FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled_features)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=["PC1_loading", "PC2_loading"],
    ).reset_index().rename(columns={"index": "variable"})
    loadings["PC1_abs_loading"] = loadings["PC1_loading"].abs()
    loadings["PC2_abs_loading"] = loadings["PC2_loading"].abs()
    loadings.to_csv(LOADINGS_CSV_PATH, index=False)
    with LOADINGS_TXT_PATH.open("w", encoding="utf-8") as loadings_file:
        loadings_file.write("PCA loadings for 5-variable K-means 2D graph\n")
        loadings_file.write(f"Input file: {INPUT_PATH}\n")
        loadings_file.write(f"Features: {', '.join(FEATURES)}\n")
        loadings_file.write(f"PC1 explained variance: {pca.explained_variance_ratio_[0]:.5f}\n")
        loadings_file.write(f"PC2 explained variance: {pca.explained_variance_ratio_[1]:.5f}\n")
        loadings_file.write(f"Total explained variance: {pca.explained_variance_ratio_.sum():.5f}\n\n")
        loadings_file.write(loadings.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
        loadings_file.write("\n")

    plot_data["pc1"] = coordinates[:, 0]
    plot_data["pc2"] = coordinates[:, 1]
    plot_data.to_csv(COORDINATES_PATH, index=False)
    pc1_low, pc1_high = plot_data["pc1"].quantile([0.01, 0.99])
    pc2_low, pc2_high = plot_data["pc2"].quantile([0.01, 0.99])
    no_outlier_data = plot_data[
        plot_data["pc1"].between(pc1_low, pc1_high)
        & plot_data["pc2"].between(pc2_low, pc2_high)
    ].copy()

    colors = {
        0: "#0072B2",
        1: "#009E73",
        2: "#D55E00",
        3: "#CC79A7",
    }
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6), dpi=190)
    for cluster in [0, 1, 2, 3]:
        cluster_data = plot_data[plot_data["kmeans_cluster"] == cluster]
        ax.scatter(
            cluster_data["pc1"],
            cluster_data["pc2"],
            s=9,
            alpha=0.18,
            color=colors.get(cluster, "#555555"),
            edgecolors="none",
            label=f"Cluster {cluster} (n={len(cluster_data):,})",
        )

    for cluster in sorted(plot_data["kmeans_cluster"].dropna().unique()):
        cluster = int(cluster)
        cluster_data = plot_data[plot_data["kmeans_cluster"] == cluster]
        center = cluster_data[["pc1", "pc2"]].mean()
        ax.scatter(
            center["pc1"],
            center["pc2"],
            s=120,
            marker="X",
            color=colors.get(cluster, "#555555"),
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(center["pc1"], center["pc2"], f" {cluster}", fontsize=10, fontweight="bold")

    ax.set_title("K-means Clusters Projected to Two Dimensions")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    fig.savefig(TRANSPARENT_OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    for output_path, point_alpha, point_size in [
        (NO_OUTLIERS_OUTPUT_PATH, 0.45, 12),
        (NO_OUTLIERS_TRANSPARENT_OUTPUT_PATH, 0.22, 9),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=190)
        for cluster in [0, 1, 2, 3]:
            cluster_data = no_outlier_data[no_outlier_data["kmeans_cluster"] == cluster]
            ax.scatter(
                cluster_data["pc1"],
                cluster_data["pc2"],
                s=point_size,
                alpha=point_alpha,
                color=colors.get(cluster, "#555555"),
                edgecolors="none",
                label=f"Cluster {cluster} (n={len(cluster_data):,})",
            )

        for cluster in sorted(no_outlier_data["kmeans_cluster"].dropna().unique()):
            cluster = int(cluster)
            cluster_data = no_outlier_data[no_outlier_data["kmeans_cluster"] == cluster]
            center = cluster_data[["pc1", "pc2"]].mean()
            ax.scatter(
                center["pc1"],
                center["pc2"],
                s=120,
                marker="X",
                color=colors.get(cluster, "#555555"),
                edgecolors="black",
                linewidths=0.8,
            )
            ax.text(center["pc1"], center["pc2"], f" {cluster}", fontsize=10, fontweight="bold")

        ax.set_title("K-means Clusters Projected to Two Dimensions")
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
        ax.grid(True, linewidth=0.5, alpha=0.25)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=170, sharex=True, sharey=True)
    axes = axes.ravel()
    x_limits = (plot_data["pc1"].min() - 0.5, plot_data["pc1"].max() + 0.5)
    y_limits = (plot_data["pc2"].min() - 0.5, plot_data["pc2"].max() + 0.5)
    for ax, cluster in zip(axes, sorted(plot_data["kmeans_cluster"].dropna().unique())):
        cluster = int(cluster)
        cluster_data = plot_data[plot_data["kmeans_cluster"] == cluster]
        ax.scatter(
            plot_data["pc1"],
            plot_data["pc2"],
            s=6,
            alpha=0.08,
            color="#999999",
            edgecolors="none",
        )
        ax.scatter(
            cluster_data["pc1"],
            cluster_data["pc2"],
            s=13,
            alpha=0.72,
            color=colors.get(cluster, "#555555"),
            edgecolors="none",
        )
        ax.set_title(f"Cluster {cluster} (n={len(cluster_data):,})")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.grid(True, linewidth=0.4, alpha=0.22)
    fig.supxlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    fig.supylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    fig.suptitle("K-means Clusters, One Panel per Cluster")
    fig.tight_layout()
    fig.savefig(FACET_OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote 2D cluster plot to {OUTPUT_PATH}")
    print(f"Wrote transparent 2D cluster plot to {TRANSPARENT_OUTPUT_PATH}")
    print(f"Wrote no-outliers 2D cluster plot to {NO_OUTLIERS_OUTPUT_PATH}")
    print(
        "Wrote transparent no-outliers 2D cluster plot to "
        f"{NO_OUTLIERS_TRANSPARENT_OUTPUT_PATH}"
    )
    print(f"Wrote faceted 2D cluster plot to {FACET_OUTPUT_PATH}")
    print(f"Wrote PCA coordinates to {COORDINATES_PATH}")
    print(f"Wrote PCA loadings to {LOADINGS_CSV_PATH}")
    print(f"Wrote PCA loadings summary to {LOADINGS_TXT_PATH}")
    print(f"Rows plotted after outlier trim: {len(no_outlier_data):,} of {len(plot_data):,}")
    print(f"PC1 explained variance: {explained[0]:.5f}")
    print(f"PC2 explained variance: {explained[1]:.5f}")
    print(f"Total 2D explained variance: {explained.sum():.5f}")


if __name__ == "__main__":
    main()
```

## src\write_kmeans_5var_analysis_notes.py

```python
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs/kmeans_5_variable_k4")
SUMMARY_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_summary.txt"
NOTES_PATH = OUTPUT_DIR / "kmeans_5_variable_k4_analysis_notes.md"
CLUSTERED_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_clustered.csv"
FEATURES = [
    "log_review_count",
    "avg_tip_compliments",
    "checkin_count",
    "photo_count",
    "inside_photo_share",
]


def main() -> None:
    clustered = pd.read_csv(CLUSTERED_PATH)
    cluster_summary = (
        clustered.groupby("kmeans_cluster", as_index=False)
        .agg(
            restaurant_count=("business_id", "count"),
            avg_stars=("stars", "mean"),
            avg_log_review_count=("log_review_count", "mean"),
            avg_tip_compliments=("avg_tip_compliments", "mean"),
            avg_checkin_count=("checkin_count", "mean"),
            avg_photo_count=("photo_count", "mean"),
            avg_inside_photo_share=("inside_photo_share", "mean"),
        )
        .sort_values("kmeans_cluster")
    )
    low_share = (
        clustered.assign(
            star_group=pd.cut(
                clustered["stars"],
                bins=[0, 3.0, 3.5, 5.0],
                labels=["low", "middle", "high"],
                include_lowest=True,
            )
        )
        .groupby("kmeans_cluster")["star_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    cluster_summary = cluster_summary.merge(low_share, on="kmeans_cluster", how="left")

    with NOTES_PATH.open("w", encoding="utf-8") as notes:
        notes.write("# K-means Analysis Notes\n\n")
        notes.write(
            "Course instruction context: the final paper should include at least one "
            "unsupervised learning method and explain the dataset, model, and results "
            "concisely. Treat this file as factual analysis notes and rewrite the final "
            "paper section in your own words.\n\n"
        )
        notes.write("## K-means Model Used\n\n")
        notes.write("- Code: `src/run_kmeans_silhouette_5_variables_k4.py`\n")
        notes.write("- Plot code: `src/plot_kmeans_silhouette_5var_k4_2d.py`\n")
        notes.write("- Input data: `outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv`\n")
        notes.write(f"- Clustered output: `{CLUSTERED_PATH}`\n")
        notes.write(f"- Summary output: `{SUMMARY_PATH}`\n")
        notes.write("- Number of clusters: `k = 4`\n")
        notes.write(f"- Observations used: `{len(clustered):,}`\n")
        notes.write("- Features used:\n")
        for feature in FEATURES:
            notes.write(f"  - `{feature}`\n")
        notes.write(
            "\nThis version drops `photo_file_count` because it is exactly equal to "
            "`photo_count` in this dataset. Keeping only one avoids double-weighting "
            "photo volume in the K-means distance calculation.\n\n"
        )
        notes.write("## Main Scores\n\n")
        notes.write("- Silhouette score: `0.47848`\n")
        notes.write("- Star-group majority accuracy: `0.46481`\n\n")
        notes.write("## Cluster Summary\n\n")
        notes.write("```text\n")
        notes.write(cluster_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
        notes.write("\n```")
        notes.write("\n\n")
        notes.write("## Interpretation\n\n")
        notes.write(
            "Cluster 0 is the largest and lowest-engagement group. These restaurants "
            "have low review volume, low check-ins, very few photos, and almost no "
            "inside-photo share.\n\n"
        )
        notes.write(
            "Cluster 1 is a medium-to-high engagement group with many inside photos. "
            "It has more reviews, check-ins, and photos than Cluster 0, and the highest "
            "inside-photo share.\n\n"
        )
        notes.write(
            "Cluster 2 is a small, very high-engagement group. It has the highest "
            "average review volume, check-ins, and photos by a large margin, as well as "
            "the highest average star rating.\n\n"
        )
        notes.write(
            "Cluster 3 is a small unusual group driven mainly by tip compliments. "
            "It has very high average tip compliments but relatively low check-ins and "
            "photo volume.\n\n"
        )
        notes.write("## Main Takeaway\n\n")
        notes.write(
            "The K-means model segments Philadelphia restaurants mostly by user "
            "engagement and photo behavior. The clearest pattern is that restaurants "
            "with very high review volume, check-ins, and photos form a small cluster "
            "with the highest average rating. The clusters are not pure rating "
            "categories because K-means is unsupervised and was not trained to predict "
            "ratings directly.\n\n"
        )
        notes.write("## Graph Outputs\n\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_transparent.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_no_outliers.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_facets.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_pca_loadings.txt`\n")

    print(f"Wrote K-means analysis notes to {NOTES_PATH}")


if __name__ == "__main__":
    main()
```
