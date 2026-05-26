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
