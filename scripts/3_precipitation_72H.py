"""
This script:
    1. Reads in the filtered microsoft and redcross building points from the previous script (2_precipitation_24H.py)
    2. Samples the PRISM precipitation rasters for 3 days at the locations of the building points and adds the precipitation values from each day to the dataframes as a new column (precipitation_72H)
    3. Saves the updated data as new CSV files (microsoft_points_B.csv and redcross_points_B.csv) for use in the next steps of the analysis.

Notes:

    -The raster is not included in the Github repository due to large file size but is publicly available from the PRISM Climate Group. To run, download all three rasters and update the file path accordingly.
            - Access data here: https://data.prism.oregonstate.edu/time_series/us/an/800m/ppt/daily/2024/ and downloasd the files for 2024-09-25, 2024-09-26, and 2024-09-27 (Example: prism_ppt_us_30s_20240927.zip)
            - 800 m x 800 m resolution
"""


from click import Path
import pandas as pd
import rasterio as rio
import numpy as np

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored

def read_building_data():
    file_path = DATA_DIR / "microsoft_points_A.csv"
    buildings = pd.read_csv(file_path)
    return buildings

def read_windshield_data():
    file_path = DATA_DIR / "redcross_points_A.csv"
    df = pd.read_csv(file_path)
    return df

def sample_72h_precipitation(points, raster_paths, column_name="precipitation_72H", x_col="lon", y_col="lat"):
    df = points.copy()
    xs = df[x_col].to_numpy()
    ys = df[y_col].to_numpy()
    coords = list(zip(xs, ys))

    total_precip = np.zeros(len(df), dtype=float)

    for raster_path in raster_paths:
        print(f"Sampling raster: {raster_path}")
        with rio.open(raster_path) as raster:
            samples = list(raster.sample(coords))
            values = np.array([s[0] for s in samples], dtype=float)
            total_precip += values

    df[column_name] = total_precip
    print(df.head())

    return df

def main():
    prism_rasters = [
        DATA_DIR / "prism_ppt_us_25m_20240925.tif",
        DATA_DIR / "prism_ppt_us_25m_20240926.tif",
        DATA_DIR / "prism_ppt_us_25m_20240927.tif"
    ]

    buildings = read_building_data()

    windshield_data = read_windshield_data()

    buildings_with_precip = sample_72h_precipitation(
        buildings,
        prism_rasters,
        column_name="precipitation_72H",
        x_col="lon",
        y_col="lat"
    )

    windshield_with_precip = sample_72h_precipitation(
        windshield_data,
        prism_rasters,
        column_name="precipitation_72H",
        x_col="x",
        y_col="y"
    )

    buildings_with_precip.to_csv(
        DATA_DIR / "microsoft_points_B.csv",
        index=False
    )
    windshield_with_precip.to_csv(
        DATA_DIR / "redcross_points_B.csv",
        index=False
    )

if __name__ == "__main__":
    main()