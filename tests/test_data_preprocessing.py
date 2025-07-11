from src.data_preprocessing import read_and_preprocess_data


def test_column_renaming():
    processed = read_and_preprocess_data("tests/fixtures/sample_obesity_data.csv")
    expected_columns = [
        "highcal_consumption",
        "veg_consumption",
        "main_meals_daily",
        "snack_consumption",
        "water_daily",
        "track_cal_intake",
        "physical_weekly",
        "tech_usage_daily",
        "alcohol_consumption",
        "transport_mode",
        "obesity_type",
    ]
    for col in expected_columns:
        assert col in processed.columns


def test_obesity_type_mapping():
    processed = read_and_preprocess_data("tests/fixtures/sample_obesity_data.csv")
    assert set(processed["obesity_type"]).issubset(
        {"Underweight", "Normal_weight", "Overweight", "Obesity"}
    )
