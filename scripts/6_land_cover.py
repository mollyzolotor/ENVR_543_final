"""
This script:
    1. Reads in the filtered microsoft and redcross building points from the previous script (5_impervious_surface.py)
    2. Samples the land cover rasters at the locations of the building points and adds the land cover values to the dataframes as a new column (landcover_code)
    3. Groups the land cover classes into broader categories (landcover_group) for easier analysis
    4. Saves the updated data as new CSV files (microsoft_points_E.csv and redcross_points_E.csv) for use in the next steps of the analysis.

Notes:

    -The raster is not included in the Github repository due to large file size but is publicly available from the Multi-Resolution Land Characteristics (MRLC) Consortium.
            - Access data here: https://www.mrlc.gov/data --> select Land Cover for 2024
    - The land cover classes were ultimately not used for the final model but the code is included here for completeness and potential future use
"""


from click import Path
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored
RASTER_PATH = DATA_DIR / "Annual_NLCD_LndCov_2024_CU_C1V1/Annual_NLCD_LndCov_2024_CU_C1V1.tif"

def read_csv_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    return df

def add_land_cover_data(df, raster_path, x_col="x", y_col="y", input_crs="EPSG:4326"):
    df = df.copy()

    # Make point geometries from the CSV coordinates
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=input_crs
    )

    with rio.open(raster_path) as src:
        
        # Reproject points to match raster CRS
        gdf = gdf.to_crs(src.crs)

        print("\n--- Reprojected geometry sample ---")
        print(gdf.geometry.head())

        # Sample raster at point locations
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        sampled = [val[0] for val in src.sample(coords)]

    gdf["landcover_code"] = sampled

    print("\n--- Sampled land cover values ---")
    print(gdf["landcover_code"].value_counts(dropna=False).head(20))

    # Group land cover classes into broader categories based on NLCD classification
            # [1] Water → Open water(11), 
            # [2] Forest → deciduous forest (41), evergreen forest (42), mixed forest (43)
            # [3] Natural vegetation→ shrub/scrub (52), grassland/herbaceous (71), 
            # [4]Agriculture → pasture/hay (81), cultivated crops (82)
            # [5] Wetlands → woody wetlands (90), emergent herbaceous wetlands (95), 
            # [6] Low Developed → developed, open space (21), developed, low intensity (22)
            # [7] High developed →  developed medium intensity (23), developed, high intensity (24)
            # [8] Barren land → Barren land (31)
            # [0] If not classified (N/A or glacial ice)

    conditions = [
        gdf["landcover_code"] == 11,
        gdf["landcover_code"].isin([41, 42, 43]),
        gdf["landcover_code"].isin([52, 71]),
        gdf["landcover_code"].isin([81, 82]),
        gdf["landcover_code"].isin([90, 95]),
        gdf["landcover_code"].isin([21, 22]),
        gdf["landcover_code"].isin([23, 24]),
        gdf["landcover_code"] == 31
    ]

    choices = [
        "Water",
        "Forest",
        "Natural vegetation",
        "Agriculture",
        "Wetlands",
        "Low developed",
        "High developed",
        "Barren land"
    ]

    gdf["landcover_group"] = np.select(conditions, choices, default="Other")

    print("\n--- Grouped land cover values ---")
    print(gdf["landcover_group"].value_counts(dropna=False))

    # Drop geometry before saving back to CSV
    gdf = gdf.drop(columns="geometry")

    return gdf


def main():
    buildings = read_csv_data(DATA_DIR / "microsoft_points_D.csv")

    windshield = read_csv_data(DATA_DIR / "redcross_points_D.csv")

    buildings_with_landcover = add_land_cover_data(
        buildings,
        raster_path=RASTER_PATH,
        x_col="lon",
        y_col="lat",
        input_crs="EPSG:4326"
    )

    windshield_with_landcover = add_land_cover_data(
        windshield,
        raster_path=RASTER_PATH,
        x_col="x",
        y_col="y",
        input_crs="EPSG:4326"
    )

    buildings_with_landcover.to_csv(DATA_DIR / "microsoft_points_E.csv", index=False)
    windshield_with_landcover.to_csv(DATA_DIR / "redcross_points_E.csv", index=False)


if __name__ == "__main__":
    main() 