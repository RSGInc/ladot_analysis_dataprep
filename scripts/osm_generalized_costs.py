import osmnx as ox
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
import os
import argparse
import operator
import requests
import zipfile
from osgeo import gdal
import glob
import shutil
import geopandas as gpd
from scipy.spatial import cKDTree
import pandas as pd


osm_mode = 'otf'
dem_mode = 'otf'
local_infra_data = True
local_volume_data = True
save_as = 'osm'
data_dir = '../data/'
stop_signs_fname = 'Stop_and_Yield_Signs/Stop_and_Yield_Signs.shp'
xwalk_fname = 'Crosswalks/Crosswalks.shp'
traffic_signals_fname = 'SignalizedIntersections_forCity/' + \
    'SignalizedIntersections.shp'
bikeways_fname = 'Bikeways_As_of_7302019/Bikeways_7302019.shp'
place = 'Los Angeles County, California, USA'
dem_formattable_path = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/' + \
    'Elevation/1/TIFF/n{0}w{1}/'
dem_formattable_fname = 'USGS_1_n{0}w{1}.tif'
slope_stat_breaks = [[2, 4, 6], [10]]
default_tags = ox.settings.useful_tags_path
addtl_tags = [
    'cycleway', 'cycleway:left', 'cycleway:right', 'bicycle', 'foot',
    'surface']
custom_tags = [
    'aadt', 'speed_peak:forward', 'speed_offpeak:forward',
    'speed_peak:backward', 'speed_offpeak:backward',
    'gen_cost_bike:forward:left', 'gen_cost_bike:forward:straight',
    'gen_cost_bike:forward:right', 'gen_cost_bike:backward:left',
    'gen_cost_bike:backward:straight', 'gen_cost_bike:backward:right',
    'gen_cost_ped:forward:left', 'gen_cost_ped:forward:straight',
    'gen_cost_ped:forward:right', 'gen_cost_ped:backward:left',
    'gen_cost_ped:backward:straight', 'gen_cost_ped:backward:right']
all_edge_tags = [
    tag for tag in default_tags if tag != 'est_width'] + addtl_tags

ox.config(
    useful_tags_path=all_edge_tags,
    osm_xml_way_tags=all_edge_tags + custom_tags,
    all_oneway=True)

# all no duplicate edges!!!!
ox.settings.bidirectional_network_types = []


def get_integer_bbox(nodes_gdf):
    """
    Generate absolute value integer lat/lon bounds from a dataframe of
    OSM nodes

    Args:
        nodes_gdf: a geopandas.GeoDataFrame object of OSM nodes with
            columns 'x' and 'y'

    Returns:
        Tuple of absolute value integer lat/lon bounds.
    """
    min_x = int(np.abs(np.floor(nodes_gdf['x'].min())))
    min_y = int(np.abs(np.ceil(nodes_gdf['y'].min())))
    max_x = int(np.abs(np.floor(nodes_gdf['x'].max())))
    max_y = int(np.abs(np.ceil(nodes_gdf['y'].max())))
    return min_x, min_y, max_x, max_y


def format_dem_url(x, y, dem_formattable_path=dem_formattable_path, dem_formattable_fname=dem_formattable_fname):
    """
    Construct full url of USGS DEM file.

    Args:
        x: 2-digit longitude
        y: 3-digit latitude
        dem_formattable_path: a generic formattable string defining the
            path to the DEM data on the USGS server
        dem_formattable_fname: a generic formattable string defining the
            endpoint of the DEM files on the USGS server

    Returns:
        full url of USGS DEM file.
    """
    formatted_path = dem_formattable_path.format(y, x)
    formatted_fname = dem_formattable_fname.format(y, x)
    full_url = formatted_path + formatted_fname
    return full_url


def download_save_geotiff(url, fname, data_dir=data_dir):
    """
    Download USGS GeoTIFF file from url

    Args:
        url: full url of USGS GeoTIFF file
        fname: name of geotiff file on disk
        data_dir: (relative) path to project data directory
    """
    res = requests.get(url)
    directory = os.path.join(data_dir, 'tmp')
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, fname + '.tif'), 'wb') as f:
        f.write(res.content)
    return


def download_save_unzip_dem(url, fname, data_dir=data_dir):
    """
    Downloads zipped archive of DEM data from USGS and extracts
    all files to disk

    Args:
        url: full url of the zipped USGS dem data
        fname: name of geotiff file on disk
        data_dir: (relative) path to project data directory
    """
    res = requests.get(url)
    zipped_fname = os.path.join(data_dir, fname + '.zip')

    with open(zipped_fname, 'wb') as foo:
        foo.write(res.content)

    directory = os.path.join(data_dir, fname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with zipfile.ZipFile(zipped_fname, 'r') as zip_ref:
        zip_ref.extractall(directory)

    return


def convert_adf_to_gtiff(fname, data_dir=data_dir):
    """
    Convert ArcGIS binary grid file raster data to geotiff

    Args:
        fname: name of geotiff file on disk
        data_dir: (relative) path to project data directory

    """

    in_fname = glob.glob(os.path.join(data_dir, fname, '**', 'w001001.adf'))[0]
    src_ds = gdal.Open(in_fname)
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.CreateCopy(data_dir + fname + '.tif', src_ds, 0)
    dst_ds = None
    src_ds = None

    return


def get_all_dems(min_x, min_y, max_x, max_y, dem_formattable_path, dem_formattable_fname=dem_formattable_fname):
    """
    Download all 1-arc second DEM data needed to cover
    OSM network bounds

    Args:
        min_x, min_y, max_x, max_y: min/max x/y integer absolute values
            of the lat/lon coordinates that define the boundaries
            of the OSM network
        dem_formattable_path: a generic formattable string defining the
            path to the DEM data on the USGS server
        dem_formattable_fname: a generic formattable string defining the
            endpoint of the DEM files on the USGS server

    Returns:
        The number of total files downloaded.
    """
    abs_min_x = min(min_x, max_x)
    abs_max_x = max(min_x, max_x)
    abs_min_y = min(min_y, max_y)
    abs_max_y = max(min_y, max_y)
    tot_x = abs_max_x - abs_min_x + 1
    tot_y = abs_max_y - abs_min_y + 1
    tot_files = max(tot_x, tot_y)
    it = 0
    for x in range(abs_min_x, abs_max_x + 1):
        x = str(x).zfill(3)
        for y in range(abs_min_y, abs_max_y + 1):
            y = str(y).zfill(2)
            fname = 'dem_n{0}_w{1}'.format(y, x)
            url = format_dem_url(
                x, y, dem_formattable_path, dem_formattable_fname)
            # _ = download_save_unzip_dem(url, fname, data_dir=data_dir)
            # _ = convert_adf_to_gtiff(fname, data_dir)
            _ = download_save_geotiff(url, fname, data_dir)
            it += 1
            print('Downloaded {0} of {1} DEMs and saved as GeoTIFF.'.format(
                it, tot_files))

    return tot_files


def get_mosaic(fname, data_dir=data_dir):
    """
    Combine individual GeoTIFFs into a single file

    Args:
        fname: name of geotiff file on disk
        data_dir: (relative) path to project data directory
    """
    directory = os.path.join(data_dir, 'tmp')
    all_tif_files = glob.glob(os.path.join(directory, '*.tif'))
    all_tifs = []
    for file in all_tif_files:
        tif = rasterio.open(file)
        all_tifs.append(tif)
    merged, out_trans = merge(all_tifs)
    out_meta = all_tifs[0].meta.copy()
    out_meta.update({
        "height": merged.shape[1],
        "width": merged.shape[2],
        "transform": out_trans})
    with rasterio.open(
            os.path.join(data_dir, 'tmp', fname), "w", **out_meta) as dest:
        dest.write(merged)


def reproject_geotiff(fname, data_dir=data_dir):
    """
    Takes a geotiff, reprojects it to epsg:2770, and saves new
    file to disk

    Args:
        fname: name of geotiff file on disk
        data_dir: (relative) path to project data directory

    """
    with rasterio.open(os.path.join(data_dir, 'tmp', fname)) as src:
        dst_crs = 'EPSG:2770'
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(
                os.path.join(data_dir, fname), 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

    return


def get_point_to_point_dists(edges_gdf):
    """
    Fetches pairwise euclidean distances from a dataframe
    of network edges containing lists of (x,y) coordinate pairs,
    each of which corresponds to a point in the LineString
    geometry of an edge.

    Args:
        edges_df: a geopandas.GeoDataFrame object with a column
            named 'coord_pairs' containing a list of
            consecutive (x,y) coordinate pairs.

    Returns:
        A pandas.Series object with lists of distances as its
            values.
    """

    tmp_df = edges_gdf.copy()
    tmp_df['dists'] = tmp_df['coord_pairs'].apply(
        lambda x: np.diag(cdist(x[:-1], x[1:])))

    return tmp_df['dists']


def get_slopes(edges_gdf):
    """
    Computes slopes along edge segments.

    Using vertical (z-axis) trajectories and lists of edge
    segment distances, calculates the slope along each segment
    of a LineString geometry for every edge.

    Args:
        edges_df: a geopandas.GeoDataFrame object with columns
            named 'z_trajectories' and 'z_dists'.

    Returns:
        A pandas.Series object with lists of slopes as its values.
    """

    tmp_df = edges_gdf.copy()
    tmp_df['z_diffs'] = tmp_df['z_trajectories'].apply(
        lambda x: np.diff(x))
    tmp_df['slopes'] = tmp_df['z_diffs'] / tmp_df['dists']

    return tmp_df['slopes']


def get_slope_mask(edges_gdf, lower, upper=None, direction="up"):
    """
    Generates an array of booleans that can be used to mask
    other arrays based on their position relative to user-defined
    boundaries.

    Args:
        edges_gdf: a geopandas.GeoDataFrame object with a column
            named 'slopes' containing a list of edge segment
            slopes
        lower: a numeric lower bound to use for filtering slopes
        upper: a numeric upper bound to use for filtering slopes
        direction: one of ["up", "down", "undirected"]

    Returns:
        A pandas.Series of boolean values
    """

    tmp_df = edges_gdf.copy()

    # convert bounds to percentages
    lower *= 0.01
    if upper:
        upper *= 0.01

    # for upslope stats apply ">=" to the lower bound
    # and "<" to the upper
    if direction in ["up", "undirected"]:
        lower_op = operator.ge
        upper_op = operator.lt

    # for downslope stats apply "<=" to the lower bound
    # and ">" to the upper
    elif direction == "down":
        lower = -1 * lower
        if upper:
            upper = -1 * upper
        lower_op = operator.le
        upper_op = operator.gt

    # for undirected stats use the absolute value of all slopes
    if direction == "undirected":
        tmp_df['slopes'] = np.abs(tmp_df['slopes'])

    if not upper:
        mask = tmp_df['slopes'].apply(lambda x: lower_op(x, lower))
    else:
        mask = tmp_df['slopes'].apply(lambda x: (
            lower_op(x, lower)) & (upper_op(x, upper)))

    return mask


def get_nearest_nodes_to_features(nodes_gdf, features_gdf):
    """
    Assign traffic control type column to edges, forward and backward.

    Args:
        nodes_gdf: a geopandas.GeoDataFrame object of OSM nodes
        features_gdf: a geopandas.GeoDataFrame object

    Returns:
        array of node indicies
    """
    assert nodes_gdf.crs == features_gdf.crs
    nodes_array = np.array(
        list(zip(nodes_gdf.geometry.x, nodes_gdf.geometry.y)))
    features_array = np.array(
        list(zip(features_gdf.geometry.x, features_gdf.geometry.y)))

    # nodes, e.g. intersections
    tree = cKDTree(nodes_array)

    # nearest node (e.g. intersection) to each feature (e.g. stop sign)
    dist, index = tree.query(features_array, k=1)
    valid_idx = index[~np.isinf(dist)]
    valid_dist = dist[~np.isinf(dist)]
    features_gdf['nn'] = valid_idx
    features_gdf['nn_dist'] = valid_dist

    # retain only those node assignments where feature is the closest feature
    # to a node (one-to-one)
    nn_idx = features_gdf.groupby('nn')['nn_dist'].idxmin().values
    nn = features_gdf.loc[nn_idx]

    # return indices of nodes (e.g. intersections) that are the closest node
    # to a feature
    result = nodes_gdf.iloc[nn['nn']].index.values
    return result


def assign_traffic_control(G, nodes_gdf, edges_gdf, data_dir=data_dir, stop_signs_fname=stop_signs_fname, traffic_signals_fname=traffic_signals_fname, local_infra_data=True):
    """
    Assign traffic control type column to edges, forward and backward.

    Args:
        G : networkx multi(di)graph
        nodes_gdf : geopandas.GeoDataFrame
            object of OSM nodes with columns 'x' and 'y'
        edges_gdf : geopandas.GeoDataFrame
        data_dir : string
            (relative) path to project data directory
        stop_signs_fname : string
            name of stop sign shapefile
        traffic_signals_fname : string
            name of traffic signals shapefile
        local_infra_data : boolean
            use local infrastructure data

    Returns:
        edges geopandas.GeoDataFrame
    """
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()
    nodes['control_type'] = None
    nodes['stop'] = False
    nodes['signal'] = False

    G_simp = ox.simplify_graph(G, strict=False)  # get intersection-only nodes
    intx, _ = ox.graph_to_gdfs(G_simp)
    intx = intx.to_crs(epsg=2770)

    non_drive_hwy_types = ox.downloader.get_osm_filter(
        'drive').split('highway"!~"')[1].split('"')[0].split("|")
    drive_edges = edges[~edges['highway'].isin(non_drive_hwy_types)]
    drive_nodes = np.unique(np.concatenate(
        (drive_edges['u'].unique(), drive_edges['v'].unique())))
    intx = intx[intx.index.isin(drive_nodes)]  # only nodes on drive network

    if local_infra_data:

        stops_path = os.path.join(data_dir, stop_signs_fname)
        try:
            stops = gpd.read_file(stops_path)
        except OSError:
            raise OSError(
                "Couldn't find stop sign data. If you have it, make sure "
                "it's at {0}. If you don't have any, make sure you run the "
                "script with the '-i' flag to avoid encountering this error "
                "again.".format(stops_path))
        stops = stops.to_crs(epsg=2770)
        nodes_w_stops = get_nearest_nodes_to_features(intx, stops)
        nodes.loc[nodes_w_stops, 'stop'] = True
        nodes.loc[
            (nodes['stop']) | (nodes['highway'] == 'stop'),
            'control_type'] = 'stop'

        signals_path = os.path.join(data_dir, traffic_signals_fname)
        try:
            signals = gpd.read_file(signals_path)
        except OSError:
            raise OSError(
                "Couldn't find traffic signal data. If you have it, make sure "
                "it's in {0}. If you don't have any, make sure you run the "
                "script with the '-i' flag to avoid encountering this error "
                "again.".format(signals_path))
        signals = signals.to_crs(epsg=2770)
        nodes_w_signal = get_nearest_nodes_to_features(intx, signals)
        nodes.loc[nodes_w_signal, 'signal'] = True
        nodes.loc[
            (nodes['signal']) | (nodes['highway'] == 'traffic_signals'),
            'control_type'] = 'signal'

    else:
        nodes.loc[nodes['highway'] == 'stop', 'control_type'] = 'stop'
        nodes.loc[
            nodes['highway'] == 'traffic_signals', 'control_type'] = 'signal'

    edges['control_type:backward'] = None
    edges.loc[:, 'control_type:backward'] = nodes.loc[
        edges['u'], 'control_type'].values
    edges['control_type:forward'] = None
    edges.loc[:, 'control_type:forward'] = nodes.loc[
        edges['v'], 'control_type'].values

    return edges


def assign_bike_infra(edges_gdf, local_infra_data=True):

    if local_infra_data:

        # load bike infra shapefile
        bikeways_path = os.path.join(data_dir, bikeways_fname)
        bikeways = gpd.read_file(bikeways_path)
        if bikeways.crs != edges_gdf.crs:
            bikeways = bikeways.to_crs(edges_gdf.crs)

        # extract points from lines
        bikeways['points'] = bikeways['geometry'].apply(
            lambda x: [x.interpolate(i, normalized=True) for i in [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])

        # store in a new dataframe
        bike_points = pd.DataFrame()
        for i, row in bikeways.iterrows():
            for geom in row['points']:
                tmp = pd.DataFrame({
                    'more_bike_infra': [row['Bikeway']],
                    'OBJECTID_1': [row['OBJECTID_1']],
                    'geometry': [geom]})
                bike_points = pd.concat((bike_points, tmp), ignore_index=True)

        bike_points = gpd.GeoDataFrame(bike_points, geometry='geometry')
        bike_points.crs = bikeways.crs

        # create bounding 4m bounding box for each point
        offset = 4
        bbox = bike_points.bounds + [-offset, -offset, offset, offset]

        # intersect bounding boxes with edges and retain edges indices
        hits = bbox.apply(
            lambda row: list(edges_gdf.sindex.intersection(row)), axis=1)
        tmp = pd.DataFrame({
            # index of points table
            "pt_idx": np.repeat(hits.index, hits.apply(len)),
            # ordinal position of line - access via iloc later
            "line_i": np.concatenate(hits.values)
        })

        # merge back geometries to the table of intersections
        tmp = tmp.join(edges_gdf.reset_index(drop=True), on="line_i")
        tmp = tmp.join(
            bike_points[['more_bike_infra', 'geometry']].rename(
                columns={'geometry': 'point'}), on="pt_idx")
        tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=bike_points.crs)

        # retain only the nearest edge to edge point
        tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
        tmp = tmp.sort_values(by=["snap_dist"])
        closest = tmp.groupby("pt_idx").first()
        bike_infra = closest.groupby('line_i').agg(
            count=('more_bike_infra', 'count'),
            mode=('more_bike_infra', pd.Series.mode))
        bike_infra['mode'] = bike_infra['mode'].apply(
            lambda x: x if isinstance(x, str) else x[0])

        # retain edges only if they intersected with more than one point
        bike_infra = bike_infra[bike_infra['count'] > 1]

        # merge back to the main edges table
        edges_gdf.loc[:, 'more_bike_infra'] = None
        edges_gdf.loc[
            bike_infra.index, 'more_bike_infra'] = bike_infra['mode'].values

        bike_points.to_file(os.path.join(data_dir, "bike_points"))

    else:
        edges_gdf.loc[:, 'more_bike_infra'] = None

    edges_gdf['bike_infra'] = None

    if ('cycleway' in edges_gdf.columns) & ('bicycle' in edges_gdf.columns):
        # no bike lane
        edges_gdf.loc[
            (pd.isnull(edges_gdf['cycleway'])) |
            (edges_gdf['cycleway'] == 'no') |
            (edges_gdf['bicycle'] == 'no') &
            (edges_gdf['more_bike_infra'] is None), 'bike_infra'] = 'no'

        # shared lane
        edges_gdf.loc[
            (edges_gdf['cycleway'].str.contains('shared')) |
            (edges_gdf['more_bike_infra'].str.contains('Route')),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[
            (edges_gdf['highway'] == 'cycleway') |
            (
                (edges_gdf['highway'] == 'path') &
                (edges_gdf['bicycle'] == 'dedicated')
            ) |
            (edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    elif 'cycleway' in edges_gdf.columns:
        # no bike lane
        edges_gdf.loc[
            (pd.isnull(edges_gdf['cycleway'])) |
            (edges_gdf['cycleway'] == 'no') &
            (edges_gdf['more_bike_infra'] is None), 'bike_infra'] = 'no'

        # shared lane
        edges_gdf.loc[
            (edges_gdf['cycleway'].str.contains('shared')) |
            (edges_gdf['more_bike_infra'].str.contains('Route')),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[
            (edges_gdf['highway'] == 'cycleway') |
            (edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    elif 'bicycle' in edges_gdf.columns:
        # no bike lane
        edges_gdf.loc[
            (edges_gdf['bicycle'] == 'no') &
            (edges_gdf['more_bike_infra'] is None), 'bike_infra'] = 'no'

        # shared lane
        edges_gdf.loc[
            (edges_gdf['more_bike_infra'].str.contains('Route')),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[
            (edges_gdf['highway'] == 'cycleway') |
            (
                (edges_gdf['highway'] == 'path') &
                (edges_gdf['bicycle'] == 'dedicated')
            ) |
            (edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    else:
        # no bike lane
        edges_gdf.loc[
            (edges_gdf['more_bike_infra'] is None), 'bike_infra'] = 'no'

        # shared lane
        edges_gdf.loc[
            (edges_gdf['more_bike_infra'].str.contains('Route')),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[
            (edges_gdf['highway'] == 'cycleway') |
            (edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    return edges_gdf


def assign_ped_infra(G, nodes_gdf, edges_gdf, data_dir=data_dir, xwalk_fname=xwalk_fname, local_infra_data=True):
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()
    nodes['xwalk'] = None

    G_simp = ox.simplify_graph(G, strict=False)  # get intersection-only nodes
    intx, _ = ox.graph_to_gdfs(G_simp)
    intx = intx.to_crs(epsg=2770)

    non_drive_hwy_types = ox.downloader.get_osm_filter(
        'drive').split('highway"!~"')[1].split('"')[0].split("|")
    drive_edges = edges[~edges['highway'].isin(non_drive_hwy_types)]
    drive_nodes = np.unique(np.concatenate(
        (drive_edges['u'].unique(), drive_edges['v'].unique())))
    intx = intx[intx.index.isin(drive_nodes)]  # only nodes on drive network

    if local_infra_data:

        xwalk_path = os.path.join(data_dir, xwalk_fname)
        try:
            xwalks = gpd.read_file(xwalk_path)
        except OSError:
            raise OSError(
                "Couldn't find cross walk data. If you have it, make sure "
                "it's at {0}. If you don't have any, make sure you run the "
                "script with the '-i' flag to avoid encountering this error "
                "again.".format(stops_path))
        xwalks = xwalks.to_crs(epsg=2770)

        # signalized crosswalks
        nodes_w_sig_xwalks = get_nearest_nodes_to_features(
            intx, xwalks[xwalks['CrossType'].isin([
                'White Controlled Crosswalk', 'Yellow Controlled Crosswalk'])])
        nodes.loc[nodes_w_sig_xwalks, 'xwalk'] = 'signal'

        # unsignalized crosswalks
        nodes_w_unsig_xwalks = get_nearest_nodes_to_features(
            intx, xwalks[
                (~pd.isnull(xwalks['CrossType'])) &
                (xwalks['CrossType'].str.contains('Uncontrolled'))])
        nodes.loc[nodes_w_sig_xwalks, 'xwalk'] = 'unsig'
        nodes.loc[
            (~pd.isnull(nodes['highway'])) &
            (nodes['highway'].str.contains('crossing')), 'xwalk'] = 'unsig'

    else:

        # assume all crosswalks are unsignalized
        nodes.loc[nodes['highway'].str.contains('crossing'), 'xwalk'] = 'unsig'

    edges['xwalk:backward'] = None
    edges.loc[:, 'xwalk:backward'] = nodes.loc[edges['u'], 'xwalk'].values
    edges['xwalk:forward'] = None
    edges.loc[:, 'xwalk:forward'] = nodes.loc[edges['v'], 'xwalk'].values

    return edges


def append_gen_cost_bike(edges_gdf):
    """
    Generates directional, turn-based generalized costs using slope
    statistics and infrastructure.

    Args:
        edges_gdf: a geopandas.GeoDataFrame object with a column
            named 'slopes' containing a list of edge segment

    Returns:
        edges_gdf
    """

    # slopes
    edges_gdf['slope_penalty:forward'] = \
        (edges_gdf['up_pct_dist_2_4'] * .371) + \
        (edges_gdf['up_pct_dist_4_6'] * 1.23) + \
        (edges_gdf['up_pct_dist_6_plus'] * 3.239)

    edges_gdf['slope_penalty:backward'] = \
        (edges_gdf['down_pct_dist_2_4'] * .371) + \
        (edges_gdf['down_pct_dist_4_6'] * 1.23) + \
        (edges_gdf['down_pct_dist_6_plus'] * 3.239)

    # turns
    edges_gdf['turn_penalty'] = 0.042

    # traffic volume at unsignalized intersections
    for direction in ['forward', 'backward']:

        edges_gdf['parallel_traffic_penalty:{0}'.format(direction)] = 0
        edges_gdf.loc[
            (edges_gdf['parallel_traffic:{0}'.format(direction)] >= 10000) &
            (pd.isnull(edges_gdf['control_type:{0}'.format(direction)])),
            'parallel_traffic_penalty:{0}'.format(direction)] = .091
        edges_gdf.loc[
            (edges_gdf['parallel_traffic:{0}'.format(direction)] >= 20000) &
            (pd.isnull(edges_gdf['control_type:{0}'.format(direction)])),
            'parallel_traffic_penalty:{0}'.format(direction)] = .231

        edges_gdf['cross_traffic_penalty_ls:{0}'.format(direction)] = 0
        edges_gdf.loc[
            (edges_gdf['cross_traffic:{0}'.format(direction)] >= 5000) &
            (pd.isnull(edges_gdf['control_type:{0}'.format(direction)])),
            'cross_traffic_penalty_ls:{0}'.format(direction)] = .041
        edges_gdf.loc[
            (edges_gdf['cross_traffic:{0}'.format(direction)] >= 10000) &
            (pd.isnull(edges_gdf['control_type:{0}'.format(direction)])),
            'cross_traffic_penalty_ls:{0}'.format(direction)] = .059
        edges_gdf.loc[
            (edges_gdf['cross_traffic:{0}'.format(direction)] >= 20000) &
            (pd.isnull(edges_gdf['control_type:{0}'.format(direction)])),
            'cross_traffic_penalty_ls:{0}'.format(direction)] = .322

        edges_gdf['cross_traffic_penalty_r:{0}'.format(direction)] = 0
        edges_gdf.loc[
            (edges_gdf['cross_traffic:{0}'.format(direction)] >= 10000) &
            (pd.isnull(edges_gdf['control_type:{0}'.format(direction)])),
            'cross_traffic_penalty_r:{0}'.format(direction)] = .038

    # traffic signals
    edges_gdf['signal_penalty:forward'] = (
        edges_gdf['control_type:forward'] == 'signal') * 0.021
    edges_gdf['signal_penalty:backward'] = (
        edges_gdf['control_type:backward'] == 'signal') * 0.021

    # stop signs
    edges_gdf['stop_sign_penalty:forward'] = (
        edges_gdf['control_type:forward'] == 'stop') * 0.005
    edges_gdf['stop_sign_penalty:backward'] = (
        edges_gdf['control_type:backward'] == 'stop') * 0.005

    # no bike lane
    edges_gdf['no_bike_penalty'] = 0
    edges_gdf.loc[
        (edges_gdf['aadt'] >= 10000) &
        (edges_gdf['bike_infra'] == 'no'), 'no_bike_penalty'] = 0.368
    edges_gdf.loc[
        (edges_gdf['aadt'] >= 20000) &
        (edges_gdf['bike_infra'] == 'no'), 'no_bike_penalty'] = 1.4
    edges_gdf.loc[
        (edges_gdf['aadt'] >= 30000) &
        (edges_gdf['bike_infra'] == 'no'), 'no_bike_penalty'] = 7.157

    # bike blvd
    edges_gdf['bike_blvd_penalty'] = 0
    edges_gdf['bike_blvd_penalty'] = (
        edges_gdf['bike_infra'] == 'blvd') * -.108

    # bike path
    edges_gdf['bike_path_penalty'] = 0
    edges_gdf['bike_path_penalty'] = (
        edges_gdf['bike_infra'] == 'path') * -.016

    # compute costs
    for direc in ['forward', 'backward']:
        for turn_type in ['left', 'straight', 'right']:
            colname = 'gen_cost_bike:' + direc + ':' + turn_type

            # all turn types have slope, stop, and bike infra penalties
            weights = edges_gdf['slope_penalty:{0}'.format(direc)] + \
                edges_gdf['stop_sign_penalty:{0}'.format(direc)] + \
                edges_gdf['no_bike_penalty'] + \
                edges_gdf['bike_blvd_penalty'] + \
                edges_gdf['bike_path_penalty']

            # left turns: add traffic signal, turn penalties, parallel traffic,
            # and cross traffic for left turn
            if turn_type == 'left':
                weights += edges_gdf['signal_penalty:{0}'.format(direc)] + \
                    edges_gdf['turn_penalty'] + \
                    edges_gdf['parallel_traffic_penalty:{0}'.format(direc)] + \
                    edges_gdf['cross_traffic_penalty_ls:{0}'.format(direc)]

            # straight out of link: add signal penalty, cross traffic for
            # straight, no bike lane penalty

            elif turn_type == 'straight':
                weights += edges_gdf['signal_penalty:{0}'.format(direc)] + \
                    edges_gdf['cross_traffic_penalty_ls:{0}'.format(direc)]

            # right turns: add turn penalty, cross traffic for right turn
            else:
                weights = weights + edges_gdf['turn_penalty'] + \
                    edges_gdf['cross_traffic_penalty_r:{0}'.format(direc)]

            # generalized cost = length + length * weighted penalties
            edges_gdf[colname] = edges_gdf['length'] + \
                edges_gdf['length'] * weights

    return edges_gdf


def append_gen_cost_ped(edges_gdf):
    edges_gdf['ped_slope_penalty:forward'] = \
        edges_gdf['up_pct_dist_10_plus'] * .99
    edges_gdf['ped_slope_penalty:backward'] = \
        edges_gdf['down_pct_dist_10_plus'] * .99
    edges_gdf['unpaved_alley_penalty'] = (
        (edges_gdf['highway'] == 'alley') |
        (edges_gdf['surface'] == 'unpaved')) * .51
    edges_gdf['busy_penalty'] = (
        edges_gdf['highway'].isin([
            'tertiary', 'tertiary_link', 'secondary', 'secondary_link',
            'primary', 'primary_link', 'trunk', 'trunk_link',
            'motorway', 'motorway_link'])) * .14

    for direc in ['forward', 'backward']:

        # if left or right turn, then the pedestrian will have to cross
        # either the self-edge or the parallel edge. If either of these
        # edges have aadt between 13k and 23k and the intersection is
        # unsignalized, apply the fixed penalty for unsignalized
        # arterial crossings
        edges_gdf.loc[:, 'unsig_art_xing_penalty_lr:{0}'.format(direc)] = ((
            (
                (edges_gdf[
                    'parallel_traffic:{0}'.format(direc)] >= 13000) &
                (edges_gdf[
                    'parallel_traffic:{0}'.format(direc)] <= 23000)
            ) |
            (
                (edges_gdf['aadt'] >= 13000) &
                (edges_gdf['aadt'] <= 23000)
            )) &
            (edges_gdf['control_type:{0}'.format(direc)] != 'signal') &
            (edges_gdf['xwalk:{0}'.format(direc)] != 'signal')) * 73

        # if left or right turn, then the pedestrian will have to cross
        # either the self-edge or the parallel edge. If either of these
        # edges have aadt between 10k and 13k and there is no
        # crosswalk, apply the fixed penalty for unmarked collector
        # crossings
        edges_gdf.loc[
            :, 'unmarked_coll_xing_penalty_lr:{0}'.format(direc)] = ((
                (
                    (edges_gdf[
                        'parallel_traffic:{0}'.format(direc)] >= 10000) &
                    (edges_gdf[
                        'parallel_traffic:{0}'.format(direc)] < 13000)
                ) |
                (
                    (edges_gdf['aadt'] >= 10000) &
                    (edges_gdf['aadt'] <= 13000)
                )) &
                (pd.isnull(edges_gdf['xwalk:{0}'.format(direc)]))) * 28

        # if straight through, then the pedestrian will have to cross
        # the cross traffic edges. If cross traffic aadt is between
        # 13k and 23k and the intersection is unsignalized,
        # apply the fixed penalty for unsignalized arterial crossings
        edges_gdf.loc[:, 'unsig_art_xing_penalty_s:{0}'.format(direc)] = (
            (
                (edges_gdf[
                    'cross_traffic:{0}'.format(direc)] >= 13000) &
                (edges_gdf[
                    'cross_traffic:{0}'.format(direc)] <= 23000)) &
            (edges_gdf['control_type:{0}'.format(direc)] != 'signal') &
            (edges_gdf['xwalk:{0}'.format(direc)] != 'signal')) * 73

        # if straight through, then the pedestrian will have to cross
        # the cross traffic edges. If cross traffic aadt is between
        # 10k and 13k and there is no crosswalk, apply the fixed
        # penalty for unmarked collector crossings
        edges_gdf.loc[:, 'unmarked_coll_xing_penalty_s:{0}'.format(direc)] = (
            (
                (edges_gdf[
                    'cross_traffic:{0}'.format(direc)] >= 10000) &
                (edges_gdf[
                    'cross_traffic:{0}'.format(direc)] < 13000)
            ) &
            (pd.isnull(edges_gdf['xwalk:{0}'.format(direc)]))) * 28

    for direc in ['forward', 'backward']:
        for turn_type in ['left', 'straight', 'right']:
            colname = 'gen_cost_ped:' + direc + ':' + turn_type

            # all turn types have slope, stop, and bike infra penalties
            weighted_length = (
                edges_gdf['ped_slope_penalty:{0}'.format(direc)] +
                edges_gdf['unpaved_alley_penalty'] +
                edges_gdf['busy_penalty']) * edges_gdf['length']

            if turn_type != 'straight':

                # fixed turn cost
                weighted_length += 54

                # unsignalized arterial crossing
                weighted_length += edges_gdf[
                    'unsig_art_xing_penalty_lr:{0}'.format(direc)]

                # unmarked collector crossing
                weighted_length += edges_gdf[
                    'unmarked_coll_xing_penalty_lr:{0}'.format(direc)]

            else:

                # unsignalized arterial crossing
                weighted_length += edges_gdf[
                    'unsig_art_xing_penalty_s:{0}'.format(direc)]

                # unmarked collector crossing
                weighted_length += edges_gdf[
                    'unmarked_coll_xing_penalty_s:{0}'.format(direc)]

            # generalized cost = length + weighted length
            edges_gdf[colname] = edges_gdf['length'] + weighted_length

    return edges_gdf


def get_speeds_and_volumes(data_dir):

    df = pd.DataFrame()

    for f in glob.glob(os.path.join(
            data_dir,
            'Big Data/OneDrive_1_1-27-2020/StreetLight_OSM_Primary' +
            'Roads_AtoB/*/*sa_all.csv'), recursive=True):
        tmp = pd.read_csv(f)
        df = pd.concat((df, tmp), ignore_index=True)
    df['direction'] = 'forward'

    for f in glob.glob(os.path.join(
            data_dir,
            'Big Data/OneDrive_1_1-27-2020/StreetLight_OSM_Primary' +
            'Roads_BtoA/*/*sa_all.csv'), recursive=True):
        tmp = pd.read_csv(f)
        tmp['direction'] = 'backward'
        df = pd.concat((df, tmp), ignore_index=True)

    df = df[df['Day Type'] == '1: Weekday (M-Th)']
    speed_pivot = df[[
        'Zone ID', 'Day Part', 'Average Daily Segment Traffic (StL Volume)',
        'direction']].set_index(['Zone ID', 'direction', 'Day Part']).unstack()
    speed_pivot['speed_tod'] = '2: Peak AM (6am-10am)'
    speed_pivot.loc[speed_pivot[
        (
            'Average Daily Segment Traffic (StL Volume)',
            '2: Peak AM (6am-10am)')] < speed_pivot[
        (
            'Average Daily Segment Traffic (StL Volume)',
            '4: Peak PM (3pm-7pm)')], 'speed_tod'] = '4: Peak PM (3pm-7pm)'
    speed_pivot = speed_pivot.reset_index()
    peak_speeds = pd.merge(
        df, speed_pivot[['Zone ID', 'direction', 'speed_tod']],
        left_on=['Zone ID', 'direction', 'Day Part'],
        right_on=['Zone ID', 'direction', 'speed_tod'])
    peak_speeds = peak_speeds[[
        'Zone ID', 'direction', 'Avg Segment Speed (mph)']]
    peak_speeds.rename(
        columns={'Avg Segment Speed (mph)': 'speed'}, inplace=True)
    off_peak_speeds = df.loc[
        df['Day Part'] == '3: Mid-Day (10am-3pm)',
        ['Zone ID', 'direction', 'Avg Segment Speed (mph)']]
    off_peak_speeds.rename(
        columns={'Avg Segment Speed (mph)': 'speed'}, inplace=True)
    speeds = pd.merge(
        peak_speeds, off_peak_speeds,
        on=['Zone ID', 'direction'], suffixes=('_peak', '_offpeak'))
    aadt = df[df['Day Part'] == '0: All Day (12am-12am)']
    aadt = df.groupby('Zone ID').agg(aadt=(
        'Average Daily Segment Traffic (StL Volume)', 'sum')).reset_index()

    return speeds, aadt


def process_volumes(edges_gdf, edges_w_vol):

    edges_gdf['bearing:forward'] = edges_gdf['bearing'].values
    edges_gdf['bearing:backward'] = (edges_gdf['bearing'].values + 180) % 360

    for i, edge in tqdm(edges_gdf.iterrows(), total=len(edges_gdf)):

        node_directions = {'forward': 'v', 'backward': 'u'}
        for direction, node in node_directions.items():

            vols = edges_w_vol[(
                (edges_w_vol['u'] == edge[node]) |
                (edges_w_vol['v'] == edge[node])) &
                (edges_w_vol.index.values != i)]  # ignore the self-edge

            if len(vols) > 0:

                # bearing of other edges must be relative to flow out of the
                # intersection, so flip the bearing of the other edges if they
                # flow *towards* the reference edge
                vols.loc[vols[node] == edge[node], 'bearing'] = \
                    (vols.loc[vols[node] == edge[node], 'bearing'] + 180) % 360

                vols['bearing_diff'] = np.abs(
                    edge['bearing:{0}'.format(direction)] -
                    vols['bearing'].values)
                edges_gdf.loc[i, 'cross_traffic:{0}'.format(direction)] = \
                    vols.loc[
                        (vols['bearing_diff'] >= 30) &
                        (vols['bearing_diff'] <= 330),
                        'aadt'].sum() / 2
                edges_gdf.loc[i, 'parallel_traffic:{0}'.format(direction)] = \
                    vols.loc[
                        (vols['bearing_diff'] < 30) |
                        (vols['bearing_diff'] > 330),
                        'aadt'].sum() / 2

    return edges_gdf


if __name__ == '__main__':

    # ingest command line args
    parser = argparse.ArgumentParser(
        description='Get slope statistics for OSM network')
    parser.add_argument(
        '-o', '--osm-filename', action='store', dest='osm_fname',
        help='local OSM XML file to use instead of grabbing data on-the-fly')
    parser.add_argument(
        '-d', '--dem-filename', action='store', dest='dem_fname',
        help='local DEM file to use instead of grabbing data on-the-fly')
    parser.add_argument(
        '-p', '--place', action='store', dest='place',
        help='valid nominatim place name')
    parser.add_argument(
        '-s', '--save-as', action='store', dest='save_as',
        choices=['osm', 'shp'], help='output file type')
    parser.add_argument(
        '-i', '--infra-off', action='store_true', dest='infra_off',
        help='do not use infrastructure in generalized cost calculations')
    parser.add_argument(
        '-v', '--volume-off', action='store_true', dest='volume_off',
        help='do not use traffic volumes in generalized cost calculations')

    options = parser.parse_args()

    if options.place:
        place = options.place
    place_for_fname_str = place.split(',')[0].replace(' ', '_')
    osm_fname = '{0}.osm'.format(place_for_fname_str)
    dem_fname = '{0}.tif'.format(place_for_fname_str)
    out_fname = '{0}'.format(place_for_fname_str)

    if options.osm_fname:
        osm_mode = 'local'
        osm_fname = options.osm_fname
    else:
        osm_mode = 'otf'

    if options.dem_fname:
        dem_mode = 'local'
        dem_fname = options.dem_fname
    else:
        dem_mode = 'otf'

    if options.save_as:
        save_as = options.save_as

    if options.infra_off:
        local_infra_data = False

    if options.volume_off:
        local_volume_data = False

    assert ox.settings.all_oneway

    # load local osm data
    print('Loading OSM data...')
    if osm_mode == 'local':
        path = os.path.join(data_dir, osm_fname)
        try:
            G = ox.graph_from_file(path, simplify=False, retain_all=True)
        except OSError:
            raise OSError(
                "Couldn't find file {0}. Use the -d flag "
                "to specify a different directory if your "
                "data is somewhere other than '../data/'.".format(path))

    # or pull it from the web "on-the-fly"
    elif osm_mode == 'otf':
        G = ox.graph_from_place(
            place, network_type='all', simplify=False, retain_all=True)

    else:
        raise ValueError(
            'Must specify a valid OSM mode. See --help '
            'for more details.')
    print('Done.')

    # simplify the graph topology by removing nodes
    # that don't mark intersections. NOTE: the full
    # edge geometries will not change.
    print('Processing the graph...')

    # add edge bearings
    G = ox.add_edge_bearings(G)

    # extract nodes/edges geodataframes and project them
    # into equidistant, meters-based coordinate system
    nodes, edges = ox.graph_to_gdfs(G)
    nodes.crs = {'init': 'epsg:4326'}
    edges.crs = {'init': 'epsg:4326'}
    edges = edges.to_crs(epsg=2770)

    # process the geometries to perform calculations
    edges['coord_pairs'] = edges['geometry'].apply(lambda x: list(x.coords))
    print('Done.')

    # load speed and volume data
    print('Loading speed and traffic volume data')
    if local_volume_data:
        speeds, aadt = get_speeds_and_volumes(data_dir)
        edges = pd.merge(
            edges, aadt, left_on='osmid', right_on='Zone ID', how='left')
        edges = pd.merge(
            edges[[col for col in edges.columns if col != 'Zone ID']],
            speeds[speeds['direction'] == 'forward'],
            left_on='osmid', right_on='Zone ID', how='left')
        edges = pd.merge(
            edges[[col for col in edges.columns if col not in [
                'Zone ID', 'direction']]],
            speeds[speeds['direction'] == 'backward'],
            left_on='osmid', right_on='Zone ID', how='left',
            suffixes=(':forward', ':backward'))

        # process volume data
        edges_w_vol = edges[~pd.isnull(edges['aadt'])]
        nodes_w_vol = np.unique(np.concatenate((
            edges_w_vol['u'].unique(), edges_w_vol['v'].unique())))
        edges['cross_traffic:forward'] = 0
        edges['parallel_traffic:forward'] = 0
        edges['cross_traffic:backward'] = 0
        edges['parallel_traffic:backward'] = 0
        edges_to_compute = edges[
            (edges['u'].isin(nodes_w_vol)) | (edges['v'].isin(nodes_w_vol))]
        edges_to_compute = process_volumes(edges_to_compute, edges_w_vol)
        edges.loc[edges_to_compute.index, 'cross_traffic:forward'] = \
            edges_to_compute['cross_traffic:forward']
        edges.loc[edges_to_compute.index, 'parallel_traffic:forward'] = \
            edges_to_compute['parallel_traffic:forward']
        edges.loc[edges_to_compute.index, 'cross_traffic:backward'] = \
            edges_to_compute['cross_traffic:backward']
        edges.loc[edges_to_compute.index, 'parallel_traffic:forward'] = \
            edges_to_compute['parallel_traffic:forward']

    else:
        for col in [
                'aadt', 'speed_peak:forward', 'speed_peak:backward',
                'speed_offpeak:forward', 'speed_offpeak:backward',
                'cross_traffic:forward', 'cross_traffic:backward',
                'parallel_traffic:forward', 'parallel_traffic:backward']:
            edges[col] = None

    # assign traffic signals to intersections
    print('Assigning traffic control to intersections.')
    edges = assign_traffic_control(
        G, nodes, edges, local_infra_data=local_infra_data)
    print('Done.')

    # assign bike infrastructure designations
    print('Assigning bicycle infrastructure designations.')
    edges = assign_bike_infra(edges, local_infra_data=local_infra_data)
    print('Done.')

    # assign ped infrastructure designations
    print('Assigning pedestrian infrastructure designations.')
    edges = assign_ped_infra(
        G, nodes, edges, local_infra_data=local_infra_data)
    print('Done.')

    # load elevation data
    if dem_mode == 'otf':
        print('Downloading DEMs from USGS...this might take a while')
        integer_bbox = get_integer_bbox(nodes)
        num_files = get_all_dems(
            *integer_bbox, dem_formattable_path, dem_formattable_fname)

        if num_files > 1:
            _ = get_mosaic(dem_fname, data_dir)
        else:
            single_file = glob(os.path.join(data_dir, 'tmp', '*.tif'))[0]
            shutil.copyfile(
                single_file, os.path.join(data_dir, 'tmp', dem_fname))
        _ = reproject_geotiff(dem_fname, data_dir)

    print('Loading the DEM from disk...')
    path = os.path.join(data_dir, dem_fname)
    try:
        dem = rasterio.open(path)
    except RasterioIOError:
        print(
            "Couldn't find file {0}. Use the -d flag "
            "to specify a different directory if your "
            "data is somewhere other than '../data/'.".format(path))

    # extract elevation trajectories from DEM. This can take a while.
    print(
        'Extracting elevation trajectories for the network edges...'
        'this might take a while.')
    z_trajectories = []
    for i, edge in tqdm(edges.iterrows(), total=len(edges)):
        z_trajectories.append([x[0] for x in dem.sample(edge['coord_pairs'])])
    edges['z_trajectories'] = z_trajectories
    print('Done.')

    print('Computing LineString distances and slopes')
    # point-to-point distances within each edge LineString geometry
    edges['dists'] = get_point_to_point_dists(edges)

    # compute slopes along each of those distances
    edges['slopes'] = get_slopes(edges)
    edges['mean_abs_slope'] = edges['slopes'].apply(
        lambda x: np.mean(np.abs(x)))

    # generate up- and down-slope stats as well as undirected
    print('Generating slope statistics going...')
    for direction in ["up", "down", "undirected"]:
        print("..." + direction)

        # iterate through pairs of slope boundaries defined
        # in line 16
        for breaks in slope_stat_breaks:
            for i, lower_bound in enumerate(breaks):
                bounds = breaks[i:i + 2]

                if len(bounds) == 2:
                    upper_bound = bounds[1]
                    upper_bound_str = str(upper_bound)

                else:
                    upper_bound = None
                    upper_bound_str = 'plus'

                for stat in ["tot", "pct"]:

                    # define the new column name to store the slope
                    # stat in the edges table
                    new_colname = '{0}_{1}_dist_{2}_{3}'.format(
                        direction, stat, lower_bound, upper_bound_str)
                    custom_tags.append(new_colname)

                    mask = get_slope_mask(
                        edges, lower_bound, upper_bound, direction)

                    # multiplying the distances by the boolean mask
                    # will set all distances that correspond to slopes
                    # outside of the mask boundaries used to 0
                    masked_dists = edges['dists'] * mask

                    # sum these masked dists to get total dist within
                    # the slope bounds
                    if stat == "tot":
                        edges[new_colname] = masked_dists.apply(sum)

                    # or divide by the total edge length to get a percentage
                    elif stat == "pct":
                        edges[new_colname] = \
                            masked_dists.apply(sum) / edges['length']

    edges = edges[[col for col in edges.columns if col not in [
        'coord_pairs', 'z_trajectories', 'dists', 'slopes']]]
    print('Done.')

    # get generalized costs for bike routing
    print('Generating generalized costs for bike routing.')
    edges = append_gen_cost_bike(edges)

    # get generalized costs for ped routing
    print('Generating generalized costs for pedestrian routing.')
    edges = append_gen_cost_ped(edges)

    # project the edges back to lat/lon coordinate system
    edges = edges.to_crs(epsg=4326)

    if save_as == 'shp':
        # turn the edges back to a graph to save as shapefile
        print('Converting edges and nodes back to graph structure.')
        print('Done.')
        print('Saving graph as shapefile...')
        nodes.gdf_name = 'nodes'
        ox.save_gdf_shapefile(nodes, 'nodes', data_dir + out_fname)
        edges.gdf_name = 'edges'
        ox.save_gdf_shapefile(edges[[
            col for col in ox.settings.osm_xml_way_tags] + [
            'osmid', 'u', 'v',
            'parallel_traffic:forward', 'parallel_traffic:backward',
            'cross_traffic:forward', 'cross_traffic:backward',
            'control_type:forward', 'control_type:backward',
            'bike_infra', 'no_bike_penalty',
            'xwalk:forward', 'xwalk:backward',
            'slope_penalty:forward', 'slope_penalty:backward',
            'parallel_traffic_penalty:forward',
            'parallel_traffic_penalty:backward',
            'cross_traffic_penalty_ls:forward',
            'cross_traffic_penalty_ls:backward',
            'cross_traffic_penalty_r:forward',
            'cross_traffic_penalty_r:backward',
            'bike_path_penalty', 'bike_blvd_penalty',
            'signal_penalty:forward', 'signal_penalty:backward',
            'stop_sign_penalty:forward', 'stop_sign_penalty:backward',
            'ped_slope_penalty:forward', 'ped_slope_penalty:backward',
            'unpaved_alley_penalty', 'busy_penalty',
            'unsig_art_xing_penalty_lr:forward',
            'unsig_art_xing_penalty_lr:backward',
            'unsig_art_xing_penalty_s:forward',
            'unsig_art_xing_penalty_s:backward',
            'unmarked_coll_xing_penalty_lr:forward',
            'unmarked_coll_xing_penalty_lr:backward',
            'unmarked_coll_xing_penalty_s:forward',
            'unmarked_coll_xing_penalty_s:backward',
            'geometry']], 'edges', data_dir + out_fname)
    elif save_as == 'osm':
        print('Saving graph as OSM XML...')
        ox.save_as_osm(
            [nodes, edges], filename=out_fname + '.osm', folder=data_dir,
            merge_edges=False)
        os.system("osmconvert {0}.osm -o={0}.osm.pbf".format(
            os.path.join(data_dir, out_fname)))
        print('File now available at {0}'.format(
            os.path.join(data_dir, out_fname + '.osm.pbf')))
    else:
        raise ValueError(
            "{0} is not a valid output file type. See --help for more "
            "details.".format(save_as))

    # clear out the tmp directory
    tmp_files = glob.glob('../data/tmp/*')
    for f in tmp_files:
        os.remove(f)
