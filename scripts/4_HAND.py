"""
This script:
    1. Reads in the filtered microsoft and redcross building points from the previous script (3_precipitation_72H.py)
    2. Samples the Height Above Nearest Drainage (HAND) rasters at the locations of the building points and adds the HAND values to the dataframes as a new column (HAND_m)
    3. Saves the updated data as new CSV files (microsoft_points_C.csv and redcross_points_C.csv) for use in the next steps of the analysis.

Notes:

    -The raster is not included in the Github repository due to large file size but is publicly available from NASA Earthdata.
            - Access data here: https://www.earthdata.nasa.gov/learn/gis/storymaps/global-30-m-hand
            - 30 m x 30 m resolution
"""

from joblib.numpy_pickle import Path
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.merge import merge
import numpy as np

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored

def read_building_data():
    file_path = DATA_DIR / "microsoft_points_B.csv"
    return pd.read_csv(file_path)

def read_windshield_data():
    file_path = DATA_DIR / "redcross_points_B.csv"
    return pd.read_csv(file_path)

def add_hand(df, hand_folder, x_col="x", y_col="y", input_crs="EPSG:4326"):
    df = df.copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=input_crs
    )

    tif_files = [
        os.path.join(hand_folder, f)
        for f in os.listdir(hand_folder)
        if f.lower().endswith(".tif")
    ]

    srcs = [rio.open(fp) for fp in tif_files]


    try:
        print("\nRaster CRS:", srcs[0].crs)
        print("Raster nodata:", srcs[0].nodata)

        # Reproject points to match raster CRS
        gdf = gdf.to_crs(srcs[0].crs)

        # Merge HAND tiles
        mosaic, transform = merge(srcs)

        temp_mosaic_path = os.path.join(hand_folder, "temp_hand_mosaic.tif")

        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "count": 1
        })

        with rio.open(temp_mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)

        print("\nTemporary mosaic saved to:")
        print(temp_mosaic_path)

        # Sample HAND values at point locations
        with rio.open(temp_mosaic_path) as src:
            coords = [(geom.x, geom.y) for geom in gdf.geometry]
            sampled = [val[0] for val in src.sample(coords)]

            if src.nodata is not None:
                sampled = [np.nan if val == src.nodata else val for val in sampled]

    finally:
        for src in srcs:
            src.close()

    gdf["HAND_m"] = sampled

    return gdf.drop(columns="geometry")

def main():
    hand_folder = "/Users/mzolotor/IRMII_flood/Helene_datasets/HAND_tiles"

    print("Reading building data...")
    buildings = read_building_data()

    print("Reading windshield data...")
    windshield = read_windshield_data()

    print("Adding HAND to buildings...")
    buildings_with_hand = add_hand(
        buildings,
        hand_folder=hand_folder,
        x_col="lon",   # change to "lon" if needed
        y_col="lat",   # change to "lat" if needed
        input_crs="EPSG:4326"
    )

    print("Adding HAND to windshield points...")
    windshield_with_hand = add_hand(
        windshield,
        hand_folder=hand_folder,
        x_col="x",   # change to "lon" if needed
        y_col="y",   # change to "lat" if needed
        input_crs="EPSG:4326"
    )

    buildings_out = DATA_DIR / "microsoft_points_C.csv"
    windshield_out = DATA_DIR / "redcross_points_C.csv"

    buildings_with_hand.to_csv(buildings_out, index=False)
    windshield_with_hand.to_csv(windshield_out, index=False)

if __name__ == "__main__":
    main()