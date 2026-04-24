"""
This script:
    1. Reads in the filtered microsoft and redcross building points from the previous script (6_land_cover.py)
    2. Samples the elevation rasters at the locations of the building points and adds the elevation values to the dataframes as a new column (elevation)
    3. Saves the updated data as new CSV files (microsoft_points_F.csv and redcross_points_F.csv) for use in the next steps of the analysis.

Notes:

    -The raster is not included in the Github repository due to large file size but is publicly available from NC One Map
            - Access data here: https://www.nconemap.gov/#directdatadownloads --> select "Elevation (DEM)" and download the DEMs for each county
    - Note that the DEMs are organized by county, so the code processes each county separately and then combines the outputs at the end. This was done to manage memory usage and processing time. Only a few counties were downloaded at a time, and then deleted. The end of the script includes a section which is commented out that combines temporary files into the final output. Run this after all county batches have been processed and the temporary files are saved in the output folder.
"""


from click import Path
import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored

def add_elevation_for_one_county(df, county_name, raster_folder, county_col,
                             x_col, y_col, input_crs="EPSG:4326"):
    county_df = df[df[county_col].str.lower() == county_name.lower()].copy()

    raster_path = os.path.join(
        raster_folder,
        f"{county_name}-DEM03",
        f"{county_name}-DEM03.tif"
    )

    print(f"\nProcessing county: {county_name}")
    print(f"Raster path: {raster_path}")

    gdf = gpd.GeoDataFrame(
        county_df,
        geometry=gpd.points_from_xy(county_df[x_col], county_df[y_col]),
        crs=input_crs
    )

    with rio.open(raster_path) as src:

        gdf = gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        sampled = [val[0] for val in src.sample(coords)]

        if src.nodata is not None:
            sampled = [np.nan if val == src.nodata else val for val in sampled]

    gdf["elevation"] = sampled
    gdf = gdf.drop(columns="geometry")

    return gdf


def read_building_data():
    file_path = DATA_DIR / "microsoft_points_E.csv"
    return pd.read_csv(file_path)


def read_windshield_data():
    file_path = DATA_DIR / "redcross_points_E.csv"
    return pd.read_csv(file_path)


def process_dataset_by_county(df, counties, raster_folder, output_folder,
                              prefix, county_col, x_col, y_col,
                              input_crs="EPSG:4326"):
    os.makedirs(output_folder, exist_ok=True)

    df[county_col] = (
    df[county_col]
    .str.strip()
    .str.lower()
    .str.replace(" county", "", regex=False)
)

    processed_counties = []

    for county_name in counties:
        result = add_elevation_for_one_county(
            df=df,
            county_name=county_name,
            raster_folder=raster_folder,
            county_col=county_col,
            x_col=x_col,
            y_col=y_col,
            input_crs=input_crs
        )

        if result is not None:
            out_path = os.path.join(output_folder, f"{county_name}_{prefix}.csv")
            result.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
            processed_counties.append(county_name)

    print(f"\nFinished {prefix}. Processed counties: {processed_counties}")


def combine_county_outputs(output_folder, prefix, final_output_path):
    files = [
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.endswith(f"_{prefix}.csv")
    ]

    if len(files) == 0:
        print(f"No files found for prefix '{prefix}'")
        return

    df_list = [pd.read_csv(f) for f in files]
    combined = pd.concat(df_list, ignore_index=True)
    combined.to_csv(final_output_path, index=False)

    print(f"Combined file saved to: {final_output_path}")


def main():
    counties = [
        "alexander", "ashe", "watauga", "avery", "mitchell", "burke",
        "yancey", "mcdowell", "rutherford", "buncombe", "henderson",
        "haywood", "jackson"
    ]

    raster_folder = DATA_DIR / "DEMs"
    output_folder = DATA_DIR / "county_outputs"

    buildings = read_building_data()

    windshield = read_windshield_data()

    process_dataset_by_county(
        df=buildings,
        counties=counties,
        raster_folder=raster_folder,
        output_folder=output_folder,
        prefix="microsoft_points_E",
        county_col="County",
        x_col="lon",
        y_col="lat",
        input_crs="EPSG:4326"
    )

    process_dataset_by_county(
        df=windshield,
        counties=counties,
        raster_folder=raster_folder,
        output_folder=output_folder,
        prefix="redcross_points_E",
        county_col="county",
        x_col="x",
        y_col="y",
        input_crs="EPSG:4326"
    )

    ## Run this only after you've finished all county batches!!
    # combine_county_outputs(
    #     output_folder,
    #     "microsoft_points_F",
    #     DATA_DIR / "microsoft_points_F.csv"
    # )
    
    # combine_county_outputs(
    #     output_folder,
    #     "redcross_points_F",
    #     DATA_DIR / "redcross_points_F.csv"
    # )

    # print("\nDone.")


if __name__ == "__main__":
    main()