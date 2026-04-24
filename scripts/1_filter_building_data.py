'''
This script:
    1. Reads in Microsoft building footprint data (pre-processed in QGIS clipped to relevant counties and labeled with county names)
    2. Reads in Red Cross windshield survey data (pre-processed in QGIS clipped to relevant counties)
    3. Reads in parcel shapefiles for the relevant counties
    4. Performs a spatial join to determine which buildings fall within the same parcels as the Red Cross points. If a building falls within the same parcel as a Red Cross point, it is removed from the Microsoft dataset to avoid overlap.
    5. Saves the filtered datasets to new CSV files for use in the next steps of the analysis.

Notes:

    -The Parcel data is from the NC OneMap Open Data Portal, but is not included in the Github repository due to large file size. To run, download shapefiles and update the file paths accordingly.
            - Access data here & download relevant counties: https://www.nconemap.gov/datasets/1de3d7d828ce4813b838ddf055b40317_1/explore?location=35.173900%2C-79.919650%2C6
 
'''


from click import Path
import pandas as pd
import geopandas as gpd

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored

#List of counties restricted to the western part of NC, where the Red Cross windshield data is available.
counties = [
    "alexander", "ashe", "watauga", "avery", "mitchell", "burke",
    "yancey", "mcdowell", "rutherford", "buncombe", "henderson",
    "haywood", "jackson"
]

def read_building_data():
    return pd.read_csv(
        DATA_DIR / "microsoft_buildings_clipped_and_labeled.csv"
    )

def read_windshield_data():
    return pd.read_csv(
        DATA_DIR / "redcross_buildings_clipped.csv"
    )

def read_parcel_data():
    parcel_list = []

    for county in counties:
        path = f"{DATA_DIR}/{county}_parcels/nc_{county}_parcels_poly.shp"
        parcels = gpd.read_file(path)
        parcel_list.append(parcels)

    all_parcels = gpd.GeoDataFrame(pd.concat(parcel_list, ignore_index=True), crs=parcel_list[0].crs)
    return all_parcels

def filter_buildings(buildings_df, windshield_df, parcels):

    windshield_gdf = gpd.GeoDataFrame(
        windshield_df,
        geometry=gpd.points_from_xy(windshield_df["x"], windshield_df["y"]),
        crs="EPSG:4326"
    ).to_crs(parcels.crs)

    buildings_gdf = gpd.GeoDataFrame(
        buildings_df,
        geometry=gpd.points_from_xy(buildings_df["lon"], buildings_df["lat"]),
        crs="EPSG:4326"
    ).to_crs(parcels.crs)

    windshield_with_parcel = gpd.sjoin(
        windshield_gdf,
        parcels,
        how="left",
        predicate="intersects"
    )

    buildings_with_parcel = gpd.sjoin(
        buildings_gdf,
        parcels,
        how="left",
        predicate="intersects"
    )

    rc_parcels = windshield_with_parcel["PARNO"].dropna().unique()

    print("Number of Red Cross parcels:", len(rc_parcels))
    print("Buildings before filter:", len(buildings_with_parcel))

    buildings_filtered = buildings_with_parcel[
        ~buildings_with_parcel["PARNO"].isin(rc_parcels)
    ].copy()

    print("Buildings after filter:", len(buildings_filtered))

    overlap_check = buildings_filtered["PARNO"].isin(rc_parcels).sum()
    print("Remaining overlaps after filter:", overlap_check)

    return buildings_filtered

def main():
    buildings_df = read_building_data()
    windshield_df = read_windshield_data()
    parcels = read_parcel_data()

    buildings_filtered = filter_buildings(buildings_df, windshield_df, parcels)

    windshield_df.to_csv(
        DATA_DIR / "filtered_redcross_points.csv",
        index=False
    )
    buildings_filtered.to_csv(
        DATA_DIR / "filtered_microsoft_points.csv",
        index=False
    )

if __name__ == "__main__":
    main()