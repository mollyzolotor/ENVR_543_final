"""
This script:
    1. Reads in the filtered microsoft and redcross building points from the previous script (1_filter_building_data.py)
    2. Samples the PRISM precipitation raster at the locations of the building points and adds the precipitation values on the day Helene hit Western NC (2024-09-27) to the dataframes as a new column
    3. Saves the updated data as new CSV files (microsoft_points_A.csv and redcross_points_A.csv) for use in the next steps of the analysis.

Notes:

    -The raster is not included in the Github repository due to large file size but is publicly available from the PRISM Climate Group. To run, download the raster and update the file path accordingly.
            - Access data here: https://data.prism.oregonstate.edu/time_series/us/an/800m/ppt/daily/2024/ and downloasd the file for 2024-09-27 (prism_ppt_us_30s_20240927.zip)
            - 800 m x 800 m resolution
"""


from click import Path
import pandas as pd
import rasterio as rio

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored

def read_building_data():
    file_path = DATA_DIR / "filtered_microsoft_points.csv"
    buildings = pd.read_csv(file_path)
    return buildings

def read_windshield_data():
    file_path = DATA_DIR / "filtered_redcross_points.csv"
    df = pd.read_csv(file_path)
    return df

def read_precipitation_data(points, x_col="lon", y_col="lat"):
    raster_path = DATA_DIR / "prism_ppt_us_30s_20240927.tif" #For Github: update this to the actual path where your raster is stored
    
    with rio.open(raster_path) as precipitation:
        df = sample_raster(precipitation, points, "precipitation", x_col, y_col)
    
    return df

def sample_raster(raster, points, column_name, x_col="lon", y_col="lat"):
    df = points.copy()
    xs = df[x_col].to_numpy()
    ys = df[y_col].to_numpy()
    coords = list(zip(xs, ys))

    samples = list(raster.sample(coords))
    values = [s[0] for s in samples]
    df[column_name] = values

    print(df.head())

    return df

def main():
    buildings = read_building_data()

    windshield_data = read_windshield_data()

    buildings_with_precip = read_precipitation_data(buildings, x_col="lon", y_col="lat")

    windshield_with_precip = read_precipitation_data(windshield_data, x_col="x", y_col="y")

    buildings_with_precip.to_csv(
        DATA_DIR / "microsoft_points_A.csv",
        index=False
    )
    windshield_with_precip.to_csv(
        DATA_DIR / "redcross_points_A.csv",
        index=False
    )

if __name__ == "__main__":
    main()