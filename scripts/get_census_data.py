import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

data_year = 2017


def get_demographic_data(data_year, census_api_key):

    base_url = "https://api.census.gov/data/{0}/acs/acs5?" + \
        "get=B01003_001E,B11001_001E&" + \
        "for=block%20group:*&in=state:06%20county:037&key={1}"

    res = requests.get(
        base_url.format(data_year, census_api_key))

    tmp = pd.DataFrame(res.json())
    df = pd.DataFrame(tmp[1:])
    df.columns = tmp.iloc[0]
    df.rename(columns={
        'tract': 'TRACT', 'block group': 'BLKGRP',
        'B01003_001E': 'population', 'B11001_001E': 'households'},
        inplace=True)

    return df


def get_geometries(data_year):

    base_url = "https://tigerweb.geo.census.gov/arcgis/rest/services/" + \
        "TIGERweb/tigerWMS_ACS{0}/MapServer/10/" + \
        "query?where=STATE%3D06+and+COUNTY%3D037&" + \
        "geometryType=esriGeometryPolygon&" + \
        "outFields=TRACT%2CBLKGRP&f=pjson&outSR=4326"

    res = requests.get(base_url.format(data_year))
    geogs = []

    features = res.json()['features']
    for i, feature in enumerate(features):
        geogs.append(feature['attributes'])
        geogs[i].update({"geometry": feature["geometry"]['rings'][0]})
    tmp = pd.DataFrame(geogs)
    tmp['geometry'] = tmp['geometry'].apply(Polygon)
    df = gpd.GeoDataFrame(tmp, geometry="geometry")
    df.crs = "EPSG:4326"

    return df


if __name__ == '__main__':

    census_api_key = os.getenv("CENSUS_API")
    demog = get_demographic_data(data_year, census_api_key)
    demog['FIPS'] = demog['state'] + demog['county'] + \
        demog['TRACT'] + demog['BLKGRP']
    geog = get_geometries(data_year)

    merged = geog.merge(demog, on=['TRACT', 'BLKGRP'])
    merged['population'] = merged['population'].astype(float)
    merged['households'] = merged['households'].astype(float)
    merged.to_file("../data/census_data")
