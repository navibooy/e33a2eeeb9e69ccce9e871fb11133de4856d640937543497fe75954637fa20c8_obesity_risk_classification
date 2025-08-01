import os
import pandas as pd

def read_and_preprocess_data(input_path, output_path=None):
    """
    Loads and preprocesses the obesity dataset.

    Steps include:
    - Handling missing values for 'FAF' and 'TUE'
    - Renaming columns to more intuitive names
    - Calculating BMI and body fat percentage
    - Mapping obesity type into 4 broad categories
    - Optionally saving the cleaned dataset to output_path

    Parameters:
        input_path (str): Path to the raw CSV file
        output_path (str, optional): If provided, the preprocessed CSV is saved to this path

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for feature engineering
    """
    data_df = pd.read_csv(input_path)
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

    data_df["BMI"] = data_df["Weight"] / (data_df["Height"] ** 2)
    data_df["BodyFat_Percentage"] = data_df.apply(
        lambda row: 1.2 * row["BMI"] + 0.23 * row["Age"] - 16.2
        if row["Gender"] == "Male"
        else 1.2 * row["BMI"] + 0.23 * row["Age"] - 5.4,
        axis=1,
    )

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

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data_df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to: {output_path}")

    return data_df
