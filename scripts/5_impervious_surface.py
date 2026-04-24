"""
This script:
    1. Reads in the filtered microsoft and redcross building points from the previous script (4_HAND.py)
    2. Samples the impervious surface rasters at the locations of the building points and adds the impervious surface values to the dataframes as a new column (impervious_surface)
    3. Saves the updated data as new CSV files (microsoft_points_D.csv and redcross_points_D.csv) for use in the next steps of the analysis.

Notes:

    -The raster is not included in the Github repository due to large file size but is publicly available from the Multi-Resolution Land Characteristics (MRLC) Consortium.
            - Access data here: https://www.mrlc.gov/data --> select Fractional Impervious Surface for 2024
"""


from click import Path
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored
RASTER_PATH = DATA_DIR / "Annual_NLCD_FctImp_2024_CU_c1v1.tif"


def read_csv_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return df


def add_impervious_surface(df, raster_path, x_col="x", y_col="y", input_crs="EPSG:4326"):
    df = df.copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=input_crs
    )

    with rio.open(raster_path) as src:

        # Reproject points to match raster CRS
        gdf = gdf.to_crs(src.crs)

        # Sample raster at point locations
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        sampled = [val[0] for val in src.sample(coords)]

        if src.nodata is not None:
            sampled = [np.nan if val == src.nodata else val for val in sampled]

    gdf["impervious_surface"] = sampled

    print("\n--- Sampled impervious surface values ---")
    print(gdf["impervious_surface"].describe())

    gdf = gdf.drop(columns="geometry")

    return gdf


def main():
    buildings = read_csv_data(DATA_DIR / "microsoft_points_C.csv")

    windshield = read_csv_data(DATA_DIR / "redcross_points_C.csv")

    buildings_with_impervious = add_impervious_surface(
        buildings,
        raster_path=RASTER_PATH,
        x_col="lon",
        y_col="lat",
        input_crs="EPSG:4326"
    )
    windshield_with_impervious = add_impervious_surface(
        windshield,
        raster_path=RASTER_PATH,
        x_col="x",
        y_col="y",
        input_crs="EPSG:4326"
    )

    buildings_with_impervious.to_csv(DATA_DIR / "microsoft_points_D.csv", index=False)
    windshield_with_impervious.to_csv(DATA_DIR / "redcross_points_D.csv", index=False)


if __name__ == "__main__":
    main() 