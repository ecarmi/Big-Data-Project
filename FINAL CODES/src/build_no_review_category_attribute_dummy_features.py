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
