import os

import pandas as pd


def read_and_preprocess_data(path):
    """
    Loads and preprocesses the obesity dataset.

    Steps include:
    - Handling missing values for 'FAF' and 'TUE'
    - Renaming columns to more intuitive names
    - Calculating BMI and body fat percentage
    - Mapping obesity type into 4 broad categories
    - Saving the cleaned dataset to 'data/preprocessed/'

    Parameters:
        path (str): Path to the raw CSV file

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for feature engineering
    """
    data_df = pd.read_csv(path)
    data_df[["FAF", "TUE"]] = data_df[["FAF", "TUE"]].apply(
        pd.to_numeric, errors="coerce"
    )
    data_df["FAF"].fillna(0, inplace=True)
    data_df["TUE"].fillna(0, inplace=True)

    new_column_names = {
        "FAVC": "highcal_consumption",
        "FCVC": "veg_consumption",
        "NCP": "main_meals_daily",
        "CAEC": "snack_consumption",
        "CH2O": "water_daily",
        "SCC": "track_cal_intake",
        "FAF": "physical_weekly",
        "TUE": "tech_usage_daily",
        "CALC": "alcohol_consumption",
        "MTRANS": "transport_mode",
        "NObeyesdad": "obesity_type",
    }
    data_df.rename(columns=new_column_names, inplace=True)

    # Add BMI and BodyFat Percentage column based on specific features
    data_df["BMI"] = data_df["Weight"] / (data_df["Height"] ** 2)
    data_df["BodyFat_Percentage"] = data_df.apply(
        lambda row: 1.2 * row["BMI"] + 0.23 * row["Age"] - 16.2
        if row["Gender"] == "Male"
        else 1.2 * row["BMI"] + 0.23 * row["Age"] - 5.4,
        axis=1,
    )

    # Define the mapping for the obesity_type column
    obesity_type_mapping = {
        "Insufficient_Weight": "Underweight",
        "Normal_Weight": "Normal_weight",
        "Overweight_Level_I": "Overweight",
        "Overweight_Level_II": "Overweight",
        "Obesity_Type_I": "Obesity",
        "Obesity_Type_II": "Obesity",
        "Obesity_Type_III": "Obesity",
    }

    data_df["obesity_type"] = data_df["obesity_type"].str.strip()
    data_df["obesity_type"] = data_df["obesity_type"].replace(obesity_type_mapping)
    preprocessed_data = data_df

    # Save to preprocessed directory
    output_path = "data/preprocessed/obesity_data_preprocessed.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preprocessed_data.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data successfully saved to {output_path}")

    return preprocessed_data
