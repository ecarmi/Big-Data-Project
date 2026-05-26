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
