# Hurricane Helene Flood Damage Modeling in Western North Carolina

This project uses spatial data and machine learning to model building-level flood damage from Hurricane Helene in Western North Carolina. The workflow combines training and validation data from Red Cross windshield survey damage observations, and precipitation, terrain, hydrologic, land cover, and impervious surface variables to predict flood damage presence and severity and building points in thirteen Western North Carolina counties. The predicted building damage is aggregated and averaged across each county, and plotted against change in economic vitality index for each county. This is used to suggest counties with high predicted damage and low economic recovery, which may be useful is aid allocation.

## Project Goals

1. Build a binary model to predict whether a building experienced damage.
2. Build a severity model to estimate damage intensity.
3. Apply trained models to Microsoft building points.
4. Compare predicted physical damage with county-level economic impact using the Economic Vitality Index.

## Data Sources

Data and trained models are not included in this repository due to size constraints. However, all raw data sources are publicly available and linked below.

Data used in the workflow include:

- Microsoft building footprint points: https://planetarycomputer.microsoft.com/dataset/ms-buildings
- Red Cross windshield survey damage data: https://www.arcgis.com/apps/dashboards/97f82695a21641e295556caec536b70a
- NC One Map parcels:nconemap.gov/datasets/nconemap::north-carolina-parcels-polygons
- PRISM precipitation data: https://prism.oregonstate.edu/downloads/
- HAND raster data: https://www.earthdata.nasa.gov/learn/gis/storymaps/global-30-m-hand
- NC One Map Digital Elevation Models: https://www.nconemap.gov/content/0ad98f623b2a445eab81f593379c1216/about
- NHDPlus flowlines: https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus
- NLCD land cover: https://www.mrlc.gov/data
- NLCD fractional impervious surface: https://www.mrlc.gov/data
- County Economic Vitality Index data

## Output Data Location

We have included several processed databases in a one drive folder that all unc affiliates can access here: https://adminliveunc-my.sharepoint.com/:f:/g/personal/rvanness_ad_unc_edu/IgBX1YdTknQsRqKNVrBLWKDDAXM2_g2HrhG7zYX3QRR1kjQ?e=LdteME

It includes the following csv files used as indicated in the workflow in scripts 9, 10, 11
  2. microsoft_points_H.csv
  3. redcoss_points_processed.csv

The final outputs are also included (and used for analysis in script 12)
  1. microsoft_points_predicted.csv
        This csv is our model output, where each point is predicted as having sustained no damage, low, moderate, or sever          damage.
  2. all_points_real_and_predicted.csv
        This csv combines the damage level at points both real and predicted to be used in analysis.
  

## Repository Structure

```
ENVR_543_final_project/
├── scripts/
│   ├── 1_filter_building_data.py
│   ├── 2_precipitation_24H.py
│   ├── 3_precipitation_72H.py
│   ├── 4_HAND.py
│   ├── 5_impervious_surface.py
│   ├── 6_land_cover.py
│   ├── 7_elevation.py
│   ├── 8_slope.py
│   ├── 9_train_risk_model_binary.py
│   ├── 10_train_risk_model_classes.py
│   ├── 11_apply_model.py
│   └── 12_economic_analysis.py
|
├── .gitignore
└── README.md