import osmnx as ox
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm
import numpy as np
import os
import argparse
import glob
import geopandas as gpd
from scipy.spatial import cKDTree
import pandas as pd

from open_elevation_profiles import open_elevation_profiles


# default vars
place = 'Los Angeles County, California, USA'
osm_mode = 'otf'
dem_mode = 'otf'
local_infra_data = True
local_volume_data = True
gen_costs_on = True
save_as = 'pbf'
slope_stat_breaks = [[2, 4, 6], [10]]
local_crs = 'EPSG:2770'

# default files, filepaths, and URLs
data_dir = './data/'
stop_signs_fname = 'Stop_and_Yield_Signs/Stop_and_Yield_Signs.shp'
xwalk_fname = 'Crosswalks/Crosswalks.shp'
traffic_signals_fname = (
    'SignalizedIntersections_forCity/'
    'SignalizedIntersections.shp')
bikeways_fname = 'Bikeways_As_of_7302019/Bikeways_7302019.shp'
streetlight_data_dir = 'Big Data/OneDrive_1_1-27-2020/'
streetlight_data_forward = 'StreetLight_OSM_PrimaryRoads_AtoB/'
streetlight_data_backward = 'StreetLight_OSM_PrimaryRoads_AtoB/'
streetlight_data_glob = '*/*sa_all.csv'

# default osmnx settings
default_tags = ox.settings.useful_tags_path
addtl_tags = [
    'cycleway', 'cycleway:left', 'cycleway:right', 'bicycle', 'foot',
    'surface']
custom_tags = [
    'speed_peak:forward',
    'speed_peak:backward',
    'speed_offpeak:forward',
    'speed_offpeak:backward',
    'slope_1:forward',
    'slope_2:forward',
    'slope_3:forward',
    'slope_4:forward',
    'slope_1:backward',
    'slope_2:backward',
    'slope_3:backward',
    'slope_4:backward',
    'self_aadt',
    'cross_aadt:forward',
    'cross_aadt:backward',
    'parallel_aadt:forward',
    'parallel_aadt:backward',
    'control_type:forward',
    'control_type:backward',
    'bike_infra:forward',
    'bike_infra:backward',
    'unpaved_alley',
    'busy',
    'xwalk:forward',
    'xwalk:backward']
all_edge_tags = [
    tag for tag in default_tags if tag != 'est_width'] + addtl_tags
ox.config(
    useful_tags_path=all_edge_tags,
    osm_xml_way_tags=all_edge_tags + custom_tags,
    all_oneway=True)
ox.settings.bidirectional_network_types = []


# user-defined functions

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


def get_nearest_nodes_to_features(nodes_gdf, features_gdf):
    """
    Perform a nearest neighbor analysis between street nodes and
    network features.

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
    features_gdf.loc[:, 'nn'] = valid_idx
    features_gdf.loc[:, 'nn_dist'] = valid_dist

    # retain only those node assignments where feature is the closest feature
    # to a node (one-to-one)
    nn_idx = features_gdf.groupby('nn')['nn_dist'].idxmin().values
    nn = features_gdf.loc[nn_idx]

    # return indices of nodes (e.g. intersections) that are the closest node
    # to a feature
    result = nodes_gdf.iloc[nn['nn']].index.values
    return result


def assign_traffic_control(
        G, nodes_gdf, edges_gdf, data_dir=data_dir,
        stop_signs_fname=stop_signs_fname,
        traffic_signals_fname=traffic_signals_fname, local_infra_data=True):
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
    intx = intx.to_crs(local_crs)

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
        stops = stops.to_crs(local_crs)
        stops = stops[~stops['TOOLTIP'].str.contains('Yield', na=False)]
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
        signals = signals.to_crs(local_crs)
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
    """
    Matches external bicycle infrastructure data to OSM
    network edges.


    Args:
        edges_gdf : geopandas.GeoDataFrame
        local_infra_data : boolean
            use local infrastructure data

    Returns:
        edges geopandas.GeoDataFrame

    """

    # assign bike infra data from external (non-OSM) data sources.
    # we will call these attributes "more_bike_infra" to distinguish
    # from the primary OSM-derived attributes
    if local_infra_data:

        # load bike infra shapefile
        bikeways_path = os.path.join(data_dir, bikeways_fname)
        bikeways = gpd.read_file(bikeways_path)
        if bikeways.crs != edges_gdf.crs:
            bikeways = bikeways.to_crs(edges_gdf.crs)

        # extract points from lines at intervals  of 5m
        # NOTE: This process can take ~2 hours to run. If you don't
        # care as much about 100% accuracy of bike matching, you can
        # replace vertex redistribution with the interpolation that is
        # commented out below. This will just create 9 points for each
        # edges instead of sampling every 5m.

        bikeways['points'] = bikeways['geometry'].apply(
            lambda x: ox.redistribute_vertices(x, 5))

        # bikeways['points'] = bikeways['geometry'].apply(
        #     lambda x: [x.interpolate(i, normalized=True) for i in [
        #         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])

        # store points in a new dataframe.
        bike_points = pd.DataFrame()
        for i, row in tqdm(bikeways.iterrows(), total=len(bikeways)):
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

        # intersect each bounding boxes with network edges and retain
        # indices of intersecting edges
        hits = bbox.apply(
            lambda row: list(edges_gdf.sindex.intersection(row)), axis=1)

        # create temp dataframe with one row for each point/edge hit
        # combination (each point may have multiple hits)
        tmp = pd.DataFrame({

            # index of points table
            "pt_idx": np.repeat(hits.index, hits.apply(len)),

            # ordinal position of edge - access via iloc later
            "line_i": np.concatenate(hits.values)
        })

        # merge back edge geometries to the table of bike points
        tmp = tmp.join(edges_gdf.reset_index(drop=True), on="line_i")
        tmp = tmp.join(
            bike_points[['more_bike_infra', 'geometry']].rename(
                columns={'geometry': 'point'}), on="pt_idx")
        tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=bike_points.crs)

        # retain only the nearest edge to each point.
        tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
        tmp = tmp.sort_values(by=["snap_dist"])
        closest = tmp.groupby("pt_idx").first()
        bike_infra = closest.groupby('line_i').agg(
            count=('more_bike_infra', 'count'),
            mode=('more_bike_infra', pd.Series.mode))

        # now each point maps to one edge, but a given edge may be
        # associated with multiple points. to assign infra characteristics
        # to that edge we just take the mode.
        bike_infra['mode'] = bike_infra['mode'].apply(
            lambda x: x if isinstance(x, str) else x[0])

        # retain edges only if they intersected with more than one point
        bike_infra = bike_infra[bike_infra['count'] > 1]

        # add new bike infra attributes to the main edges table
        edges_gdf.loc[:, 'more_bike_infra'] = None
        edges_gdf.loc[
            bike_infra.index, 'more_bike_infra'] = bike_infra['mode'].values

        # save the bike points to disk for external data validation
        # bike_points.to_file(os.path.join(data_dir, "bike_points"))

    else:
        edges_gdf.loc[:, 'more_bike_infra'] = None

    # the main "bike_infra" attributes come directly from OSM.
    edges_gdf['bike_infra'] = None

    if ('cycleway' in edges_gdf.columns) & ('bicycle' in edges_gdf.columns):

        # shared lane
        edges_gdf.loc[(
            edges_gdf['cycleway'].str.contains('shared', na=False)) | (
            edges_gdf['more_bike_infra'].str.contains('Route', na=False)),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[(
            edges_gdf['highway'] == 'cycleway') | ((
                edges_gdf['highway'] == 'path') & (
                edges_gdf['bicycle'] == 'dedicated')) | (
            edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    elif 'cycleway' in edges_gdf.columns:

        # shared lane
        edges_gdf.loc[(
            edges_gdf['cycleway'].str.contains('shared', na=False)) | (
            edges_gdf['more_bike_infra'].str.contains('Route', na=False)),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[(edges_gdf['highway'] == 'cycleway') | (
            edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    elif 'bicycle' in edges_gdf.columns:

        # shared lane
        edges_gdf.loc[
            (edges_gdf['more_bike_infra'].str.contains('Route', na=False)),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[(
            edges_gdf['highway'] == 'cycleway') | ((
                edges_gdf['highway'] == 'path') & (
                edges_gdf['bicycle'] == 'dedicated')) | (
            edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    else:

        # shared lane
        edges_gdf.loc[
            (edges_gdf['more_bike_infra'].str.contains('Route', na=False)),
            'bike_infra'] = 'blvd'

        # bike path
        edges_gdf.loc[(
            edges_gdf['highway'] == 'cycleway') | (
            edges_gdf['more_bike_infra'] == 'Path'), 'bike_infra'] = 'path'

    return edges_gdf


def assign_ped_infra(
        G, nodes_gdf, edges_gdf, data_dir=data_dir,
        xwalk_fname=xwalk_fname, local_infra_data=True):
    """
    Assigns pedestrian infrastructure types to network edges

    Args:
        G : networkx multi(di)graph
        nodes_gdf : geopandas.GeoDataFrame
            object of OSM nodes with columns 'x' and 'y'
        edges_gdf : geopandas.GeoDataFrame
        data_dir : string
            (relative) path to project data directory
        xwalk_fname : string
            name of crosswalk shapefile
        traffic_signals_fname : string
            name of traffic signals shapefile
        local_infra_data : boolean
            use local infrastructure data

    Returns:
        edges geopandas.GeoDataFrame
    """
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()
    nodes['xwalk'] = None

    G_simp = ox.simplify_graph(G, strict=False)  # get intersection-only nodes
    intx, _ = ox.graph_to_gdfs(G_simp)
    intx = intx.to_crs(local_crs)

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
                "again.".format(xwalk_path))
        xwalks = xwalks.to_crs(local_crs)

        # signalized crosswalks
        nodes_w_sig_xwalks = get_nearest_nodes_to_features(
            intx, xwalks[xwalks['CrossType'].isin([
                'White Controlled Crosswalk', 'Yellow Controlled Crosswalk'])])
        nodes.loc[nodes_w_sig_xwalks, 'xwalk'] = 'signal'

        # unsignalized crosswalks
        nodes_w_unsig_xwalks = get_nearest_nodes_to_features(
            intx, xwalks[
                (xwalks['CrossType'].str.contains('Uncontrolled', na=False))])
        nodes.loc[nodes_w_unsig_xwalks, 'xwalk'] = 'unsig'
        nodes.loc[
            (nodes['highway'].str.contains('crossing', na=False)),
            'xwalk'] = 'unsig'

    else:

        # assume all crosswalks are unsignalized
        nodes.loc[
            nodes['highway'].str.contains('crossing', na=False),
            'xwalk'] = 'unsig'

    edges['xwalk:backward'] = None
    edges.loc[:, 'xwalk:backward'] = nodes.loc[edges['u'], 'xwalk'].values
    edges['xwalk:forward'] = None
    edges.loc[:, 'xwalk:forward'] = nodes.loc[edges['v'], 'xwalk'].values

    return edges


def generate_xtra_conveyal_tags(edges_gdf):
    """
    Convert attributes to generic tag names for gen cost
    computation by Conveyal.

    Args:
        edges_gdf: geopandas.GeoDataFrame object

    Returns:
        edges_gdf: geopandas.GeoDataFrame
    """

    # slopes
    edges_gdf['slope_1:forward'] = edges_gdf['up_pct_dist_2_4']
    edges_gdf['slope_2:forward'] = edges_gdf['up_pct_dist_4_6']
    edges_gdf['slope_3:forward'] = edges_gdf['up_pct_dist_6_plus']
    edges_gdf['slope_4:forward'] = edges_gdf['up_pct_dist_10_plus']
    edges_gdf['slope_1:backward'] = edges_gdf['down_pct_dist_2_4']
    edges_gdf['slope_2:backward'] = edges_gdf['down_pct_dist_4_6']
    edges_gdf['slope_3:backward'] = edges_gdf['down_pct_dist_6_plus']
    edges_gdf['slope_4:backward'] = edges_gdf['down_pct_dist_10_plus']

    # aadt
    edges_gdf['self_aadt'] = edges_gdf['aadt']
    edges_gdf['cross_aadt:forward'] = edges_gdf['cross_traffic:forward']
    edges_gdf['cross_aadt:backward'] = edges_gdf['cross_traffic:backward']
    edges_gdf['parallel_aadt:forward'] = edges_gdf['parallel_traffic:forward']
    edges_gdf[
        'parallel_aadt:backward'] = edges_gdf['parallel_traffic:backward']

    # bike infra
    edges_gdf['bike_infra:forward'] = edges_gdf['bike_infra']
    edges_gdf['bike_infra:backward'] = edges_gdf['bike_infra']
    edges_gdf['bike_infra_0:forward'] = pd.isnull(edges_gdf['bike_infra'])

    # ped infra
    edges_gdf['unpaved_alley'] = (
        edges_gdf['highway'] == 'alley') | (edges_gdf['surface'] == 'unpaved')
    edges_gdf['busy'] = edges_gdf['highway'].isin([
        'tertiary', 'tertiary_link', 'secondary', 'secondary_link',
        'primary', 'primary_link', 'trunk', 'trunk_link',
        'motorway', 'motorway_link'])

    return edges_gdf


def append_gen_cost_bike(edges_gdf):
    """
    Generates directional, turn-based generalized costs using slope
    statistics and infrastructure.

    Args:
        edges_gdf: geopandas.GeoDataFrame object

    Returns:
        edges_gdf: geopandas.GeoDataFrame
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
    edges_gdf['turn_penalty'] = 54

    # traffic volume at unsignalized intersections
    for direc in ['forward', 'backward']:

        edges_gdf['parallel_traffic_penalty:' + direc] = 0
        edges_gdf.loc[(
            edges_gdf['parallel_traffic:' + direc] >= 10000) & (
            pd.isnull(edges_gdf['control_type:' + direc])),
            'parallel_traffic_penalty:' + direc] = 117
        edges_gdf.loc[(
            edges_gdf['parallel_traffic:' + direc] >= 20000) & (
            pd.isnull(edges_gdf['control_type:' + direc])),
            'parallel_traffic_penalty:' + direc] = 297

        edges_gdf['cross_traffic_penalty_ls:' + direc] = 0
        edges_gdf.loc[(edges_gdf['cross_traffic:' + direc] >= 5000) & (
            pd.isnull(edges_gdf['control_type:' + direc])),
            'cross_traffic_penalty_ls:' + direc] = 78
        edges_gdf.loc[(edges_gdf['cross_traffic:' + direc] >= 10000) & (
            pd.isnull(edges_gdf['control_type:' + direc])),
            'cross_traffic_penalty_ls:' + direc] = 81
        edges_gdf.loc[(edges_gdf['cross_traffic:' + direc] >= 20000) & (
            pd.isnull(edges_gdf['control_type:' + direc])),
            'cross_traffic_penalty_ls:' + direc] = 424

        edges_gdf['cross_traffic_penalty_r:' + direc] = 0
        edges_gdf.loc[(edges_gdf['cross_traffic:' + direc] >= 10000) & (
            pd.isnull(edges_gdf['control_type:' + direc])),
            'cross_traffic_penalty_r:' + direc] = 50

        # traffic signals
        edges_gdf['signal_penalty:' + direc] = (
            edges_gdf['control_type:' + direc] == 'signal') * 27

        # stop signs
        edges_gdf['stop_sign_penalty:' + direc] = (
            edges_gdf['control_type:' + direc] == 'stop') * 6

    # no bike lane
    edges_gdf['no_bike_penalty'] = 0
    edges_gdf.loc[(edges_gdf['aadt'] >= 10000) & (
        pd.isnull(edges_gdf['bike_infra'])), 'no_bike_penalty'] = 0.368
    edges_gdf.loc[(
        edges_gdf['aadt'] >= 20000) & (
        pd.isnull(edges_gdf['bike_infra'])), 'no_bike_penalty'] = 1.4
    edges_gdf.loc[(
        edges_gdf['aadt'] >= 30000) & (
        pd.isnull(edges_gdf['bike_infra'])), 'no_bike_penalty'] = 7.157

    # bike blvd
    edges_gdf['bike_blvd_penalty'] = 0
    edges_gdf['bike_blvd_penalty'] = (
        edges_gdf['bike_infra'] == 'blvd') * -.108

    # bike path
    edges_gdf['bike_path_penalty'] = 0
    edges_gdf['bike_path_penalty'] = (
        edges_gdf['bike_infra'] == 'path') * -.16

    # compute costs
    for direc in ['forward', 'backward']:

        # link penalties are distance dependent so we add them up
        # and then multiply by the length of the egde
        dist_dep_penalties = edges_gdf['slope_penalty:' + direc] + \
            edges_gdf['no_bike_penalty'] + \
            edges_gdf['bike_blvd_penalty'] + \
            edges_gdf['bike_path_penalty']
        colname = 'gen_cost_bike:' + direc + ':link'
        edges_gdf[colname] = edges_gdf['length'] + \
            (edges_gdf['length'] * dist_dep_penalties)

        # turn penalties are not distance dependent so we just add up the
        # relevant penalties depending on the turn type
        for turn_type in ['left', 'straight', 'right']:
            colname = 'gen_cost_bike:' + direc + ':' + turn_type

            # all turn types have stop sign penalties
            edges_gdf[colname] = edges_gdf['stop_sign_penalty:' + direc]

            # left turns: add traffic signal, turn penalties, parallel traffic,
            # and cross traffic for left turn
            if turn_type == 'left':
                edges_gdf[colname] += edges_gdf['signal_penalty:' + direc] + \
                    edges_gdf['turn_penalty'] + \
                    edges_gdf['parallel_traffic_penalty:' + direc] + \
                    edges_gdf['cross_traffic_penalty_ls:' + direc]

            # straight out of link: add signal penalty, x-traffic for straight
            elif turn_type == 'straight':
                edges_gdf[colname] += edges_gdf['signal_penalty:' + direc] + \
                    edges_gdf['cross_traffic_penalty_ls:' + direc]

            # right turns: add turn penalty, cross traffic for right turn
            else:
                edges_gdf[colname] = edges_gdf['turn_penalty'] + \
                    edges_gdf['cross_traffic_penalty_r:' + direc]

    return edges_gdf


def append_gen_cost_ped(edges_gdf):
    """
    Generates directional, turn-based generalized costs using slope
    statistics and infrastructure.

    Args:
        edges_gdf: geopandas.GeoDataFrame object

    Returns:
        edges_gdf: geopandas.GeoDataFrame object
    """
    edges_gdf['ped_slope_penalty:forward'] = \
        edges_gdf['up_pct_dist_10_plus'] * .99
    edges_gdf['ped_slope_penalty:backward'] = \
        edges_gdf['down_pct_dist_10_plus'] * .99
    edges_gdf['unpaved_alley_penalty'] = ((
        edges_gdf['highway'] == 'alley') | (
        edges_gdf['surface'] == 'unpaved')) * .51
    edges_gdf['busy_penalty'] = (
        edges_gdf['highway'].isin([
            'tertiary', 'tertiary_link', 'secondary', 'secondary_link',
            'primary', 'primary_link', 'trunk', 'trunk_link',
            'motorway', 'motorway_link'])) * .14

    # compute turn-based penalties
    for direc in ['forward', 'backward']:

        # if left or right turn, then the pedestrian will have to cross
        # either the self-edge or the parallel edge. If either of these
        # edges have aadt between 13k and 23k and the intersection is
        # unsignalized, apply the fixed penalty for unsignalized
        # arterial crossings
        edges_gdf.loc[:, 'unsig_art_xing_penalty_lr:{0}'.format(direc)] = (((
            edges_gdf['parallel_traffic:{0}'.format(direc)] >= 13000) | (
            edges_gdf['aadt'] >= 13000)) & (
            edges_gdf['control_type:{0}'.format(direc)] != 'signal') & (
            edges_gdf['xwalk:{0}'.format(direc)] != 'signal')) * 73

        # if left or right turn, then the pedestrian will have to cross
        # either the self-edge or the parallel edge. If either of these
        # edges have aadt between 10k and 13k and there is no
        # crosswalk, apply the fixed penalty for unmarked collector
        # crossings
        edges_gdf.loc[:, 'unmarked_coll_xing_penalty_lr:{0}'.format(direc)] = (
            (((
                edges_gdf['parallel_traffic:{0}'.format(direc)] >= 10000) & (
                edges_gdf['parallel_traffic:{0}'.format(direc)] < 13000)) | ((
                    edges_gdf['aadt'] >= 10000) & (
                    edges_gdf['aadt'] < 13000))) & (
                pd.isnull(edges_gdf['xwalk:{0}'.format(direc)]))) * 28

        # if straight through, then the pedestrian will have to cross
        # the cross traffic edges. If cross traffic aadt is greater than
        # 13k and the intersection is unsignalized,
        # apply the fixed penalty for unsignalized arterial crossings
        edges_gdf.loc[:, 'unsig_art_xing_penalty_s:{0}'.format(direc)] = ((
            edges_gdf['cross_traffic:{0}'.format(direc)] >= 13000) & (
            edges_gdf['control_type:{0}'.format(direc)] != 'signal') & (
            edges_gdf['xwalk:{0}'.format(direc)] != 'signal')) * 73

        # if straight through, then the pedestrian will have to cross
        # the cross traffic edges. If cross traffic aadt is between
        # 10k and 13k and there is no crosswalk, apply the fixed
        # penalty for unmarked collector crossings
        edges_gdf.loc[:, 'unmarked_coll_xing_penalty_s:{0}'.format(direc)] = ((
            (
                edges_gdf['cross_traffic:{0}'.format(direc)] >= 10000) & (
                edges_gdf['cross_traffic:{0}'.format(direc)] < 13000)) & (
            pd.isnull(edges_gdf['xwalk:{0}'.format(direc)]))) * 28

    for direc in ['forward', 'backward']:

        # link penalties are distance dependent so we add them up
        # and then multiply by the length of the egde
        colname = 'gen_cost_ped:' + direc + ':link'
        dist_dep_penalties = edges_gdf[
            'ped_slope_penalty:{0}'.format(direc)] + \
            edges_gdf['unpaved_alley_penalty'] + \
            edges_gdf['busy_penalty']
        edges_gdf[colname] = edges_gdf['length'] + \
            (edges_gdf['length'] * dist_dep_penalties)

        for turn_type in ['left', 'straight', 'right']:
            colname = 'gen_cost_ped:' + direc + ':' + turn_type

            # turn penalties are not distance dependent so we just add up
            # the relevant columns

            if turn_type != 'straight':

                # fixed turn cost + unsignalized arterial crossing +
                # unmarked collector crossing
                edges[colname] = 54 + \
                    edges_gdf['unsig_art_xing_penalty_lr:' + direc] + \
                    edges_gdf['unmarked_coll_xing_penalty_lr:' + direc]

            else:

                # unsignalized arterial crossing + unmarked collector crossing
                edges[colname] = edges_gdf[
                    'unsig_art_xing_penalty_s:' + direc] + edges_gdf[
                    'unmarked_coll_xing_penalty_s:' + direc]

    return edges_gdf


def get_speeds_and_volumes(data_dir, streetlight_data_dir):
    """
    Extract and process speed and volume data from Streetlight
    Data extracts.

    Args:
        data_dir : string
            (relative) path to project data directory

    Returns:
        speeds: geopandas.GeoDataFrame of speed data keyed on
            OSM edge IDs
        aadt: geopandas.GeoDataFrame of traffic volume data
            keyed on OSM edge IDs
    """

    streetlight_data_path = os.path.join(data_dir, streetlight_data_dir)
    forward_glob = os.path.join(
        streetlight_data_path, streetlight_data_forward,
        streetlight_data_glob)
    backward_glob = os.path.join(
        streetlight_data_path, streetlight_data_backward,
        streetlight_data_glob)

    # df to store all data
    df = pd.DataFrame()

    # forward flow volumes
    for f in glob.glob(forward_glob, recursive=True):
        tmp = pd.read_csv(f)
        tmp['direction'] = 'forward'
        df = pd.concat((df, tmp), ignore_index=True)

    # backward flow volumes
    for f in glob.glob(backward_glob, recursive=True):
        tmp = pd.read_csv(f)
        tmp['direction'] = 'backward'
        df = pd.concat((df, tmp), ignore_index=True)

    df = df[df['Day Type'] == '1: Weekday (M-Th)']

    # peak speed based on (Peak AM | Peak PM) Day Part depending
    # on which corresponds to the max traffic volume
    speed_pivot = df[[
        'Zone ID', 'Day Part', 'Average Daily Segment Traffic (StL Volume)',
        'direction']].set_index(['Zone ID', 'direction', 'Day Part']).unstack()
    speed_pivot['speed_tod'] = '2: Peak AM (6am-10am)'
    speed_pivot.loc[speed_pivot[(
        'Average Daily Segment Traffic (StL Volume)',
        '2: Peak AM (6am-10am)')] < speed_pivot[(
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

    # off-peak speeds based on midday Day Part
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


def process_volumes(edges_gdf, edges_w_vol_gdf):
    """
    Compute aggregate cross traffic and parallel traffic volumes for
    all OSM edges that connect to an edge with volume data.

    Args:
        edges_gdf: geopandas.GeoDataFrame object of all OSM edges
        edges_w_vol_gdf: geopandas.GeoDataFrame object of OSM edges but
            only those edges that have volume data

    Returns:
        edges_gdf: geopandas.GeoDataFrame object
    """

    edges_gdf.loc[:, 'bearing:forward'] = edges_gdf['bearing'].values
    edges_gdf.loc[:, 'bearing:backward'] = (
        edges_gdf['bearing'].values + 180) % 360

    for i, edge in tqdm(edges_gdf.iterrows(), total=len(edges_gdf)):

        node_directions = {'forward': 'v', 'backward': 'u'}
        for direction, node in node_directions.items():

            vols = edges_w_vol[((
                edges_w_vol['u'] == edge[node]) | (
                edges_w_vol['v'] == edge[node])) & (
                    edges_w_vol.index.values != i)]  # ignore the self-edge

            if len(vols) > 0:

                # bearing of other edges must be relative to flow out of the
                # intersection, so flip the bearing of the other edges if they
                # flow *towards* the reference edge
                vols.loc[vols[node] == edge[node], 'bearing'] = \
                    (vols.loc[vols[node] == edge[node], 'bearing'] + 180) % 360

                vols.loc[:, 'bearing_diff'] = np.abs(edge[
                    'bearing:{0}'.format(direction)] - vols['bearing'].values)
                edges_gdf.loc[i, 'cross_traffic:{0}'.format(direction)] = \
                    vols.loc[(vols['bearing_diff'] >= 30) & (
                        vols['bearing_diff'] <= 330),
                        'aadt'].sum() / 2
                edges_gdf.loc[i, 'parallel_traffic:{0}'.format(direction)] = \
                    vols.loc[(vols['bearing_diff'] < 30) | (
                        vols['bearing_diff'] > 330),
                        'aadt'].sum() / 2

    return edges_gdf


if __name__ == '__main__':

    # updated data dir with absolute path
    data_dir = os.path.abspath(data_dir)

    # ingest command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--osm-filename', action='store', dest='osm_fname',
        help='local OSM XML file to use instead of grabbing data on-the-fly')
    parser.add_argument(
        '-d', '--dem-filename', action='store', dest='dem_fname',
        help='local DEM file to use instead of grabbing data on-the-fly')
    parser.add_argument(
        '-p', '--place', action='store', dest='place',
        help='valid nominatim place name. default is {0}'.format(place))
    parser.add_argument(
        '-s', '--save-as', action='store', dest='save_as',
        choices=['osm', 'pbf', 'shp'], help='output file type')
    parser.add_argument(
        '-i', '--infra-off', action='store_true', dest='infra_off',
        help='do not use infrastructure in generalized cost calculations')
    parser.add_argument(
        '-v', '--volumes-off', action='store_true', dest='volumes_off',
        help='do not use traffic volumes in generalized cost calculations')
    parser.add_argument(
        '-x', '--no-local-data', action='store_true', dest='no_local_data',
        help='ignore all local data (infrastructure, traffic volumes, etc.)')
    parser.add_argument(
        '-g', '--gen_costs', action='store', dest='gen_costs',
        chocies=['on', 'off'], help='toggle generalized cost computation')

    options = parser.parse_args()

    # overwrite defaults with runtime args
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

    if options.volumes_off:
        local_volume_data = False

    if options.no_local_data:
        local_infra_data = False
        local_volume_data = False

    if options.gen_costs:
        try:
            gen_costs_on = {'on': True, 'off': False}[options.gen_costs]
        except KeyError:
            raise KeyError(
                "gen_cost flag misspecified. See --help for more details")

    assert ox.settings.all_oneway

    # 1. LOAD OSM DATA
    print('Loading OSM data...')

    # load from disk
    if osm_mode == 'local':
        osm_path = os.path.join(data_dir, osm_fname)
        try:
            G = ox.graph_from_file(osm_path, simplify=False, retain_all=True)
        except OSError:
            raise OSError(
                "Couldn't find file {0}. Make sure it is in "
                "the data directory ({1}).".format(osm_fname, data_dir))

    # or pull it from the web "on-the-fly"
    elif osm_mode == 'otf':
        G = ox.graph_from_place(
            place, network_type='all', simplify=False, retain_all=True)

    else:
        raise ValueError(
            'Must specify a valid OSM mode. See --help '
            'for more details.')
    print('Done.')

    # 2. GRAPH PRE-PROCESSING
    print('Processing the graph...')

    # we will need edge bearings to process AADT data, but skip
    # this step otherwise because it takes a long time to run
    if local_volume_data:
        G = ox.add_edge_bearings(G)

    # extract nodes/edges geodataframes and project them
    # into equidistant, meters-based coordinate system
    nodes, edges = ox.graph_to_gdfs(G)
    nodes.crs = 'EPSG:4326'
    edges.crs = 'EPSG:4326'
    edges = edges.to_crs(local_crs)

    # process the geometries to perform calculations
    edges['coord_pairs'] = edges['geometry'].apply(lambda x: list(x.coords))
    print('Done.')

    # 3. (OPTIONAL) LOAD SPEED AND VOLUME DATA
    if local_volume_data:

        print('Loading speed and traffic volume data')
        speeds, aadt = get_speeds_and_volumes(data_dir, streetlight_data_dir)
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

        # process volumes
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

        # add dummy cols
        for col in [
                'aadt', 'speed_peak:forward', 'speed_peak:backward',
                'speed_offpeak:forward', 'speed_offpeak:backward',
                'cross_traffic:forward', 'cross_traffic:backward',
                'parallel_traffic:forward', 'parallel_traffic:backward']:
            edges[col] = None

    # # 4. PROCESS INFRASTRUCTURE DATA
    # # assign traffic signals to intersections
    # print('Assigning traffic control to intersections.')
    # edges = assign_traffic_control(
    #     G, nodes, edges, local_infra_data=local_infra_data)
    # print('Done.')

    # # assign bike infrastructure designations
    # print('Assigning bicycle infrastructure designations.')
    # edges = assign_bike_infra(edges, local_infra_data=local_infra_data)
    # print('Done.')

    # # assign ped infrastructure designations
    # print('Assigning pedestrian infrastructure designations.')
    # edges = assign_ped_infra(
    #     G, nodes, edges, local_infra_data=local_infra_data)
    # print('Done.')

    # 5. COMPUTE SLOPE STATISTICS
    dp = open_elevation_profiles.DEMProfiler(
        data_dir=data_dir, local_crs=local_crs)

    # get dem data
    if dem_mode == 'otf':
        print('Downloading DEMs from USGS...this might take a while')
        integer_bbox = get_integer_bbox(nodes)
        dp.download_usgs_dem(integer_bbox, dem_fname)

    print('Loading the DEM from disk...')
    dem_path = os.path.join(data_dir, dem_fname)
    try:
        dem = rasterio.open(dem_path)
    except RasterioIOError:
        print(
            "Couldn't find file {0}. Make sure it is in "
            "the data directory ({1}).".format(
                osm_path, os.path.abspath(data_dir)))

    # extract elevation trajectories from DEM. This can take a while.
    print(
        'Extracting elevation trajectories for the network edges. '
        'This might take a while...')
    edges['z_trajectories'] = dp.get_z_trajectories(edges, dem)
    print('Done.')

    print('Computing LineString distances and slopes')
    # point-to-point distances within each edge LineString geometry
    edges['dists'] = dp.get_point_to_point_dists(edges)

    # compute slopes along each of those distances
    edges['slopes'] = dp.get_slopes(edges)
    edges['mean_abs_slope'] = edges['slopes'].apply(
        lambda x: np.mean(np.abs(x)))
    print('Done.')

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

                    mask = dp.get_slope_mask(
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

    # 6. (OPTIONAL) COMPUTE GENERALIZED COSTS ALONG THE EDGES
    if gen_costs_on:
        # get generalized costs for bike routing
        print('Generating generalized costs for bike routing.')
        edges = append_gen_cost_bike(edges)

        # get generalized costs for ped routing
        print('Generating generalized costs for pedestrian routing.')
        edges = append_gen_cost_ped(edges)

    # 7. STORE RESULTS TO DISK

    edges = generate_xtra_conveyal_tags(edges)

    # project the edges back to lat/lon coordinate system
    edges = edges.to_crs('EPSG:4326')

    if save_as == 'shp':
        # turn the edges back to a graph to save as shapefile
        print('Saving graph as shapefile. This might take a while...')
        nodes.gdf_name = 'nodes'
        ox.save_gdf_shapefile(nodes, 'nodes', data_dir + out_fname)
        edges.gdf_name = 'edges'
        ox.save_gdf_shapefile(
            edges[[col for col in ox.settings.osm_xml_way_tags] + [
                'osmid', 'u', 'v', 'geometry']],
            'edges', data_dir + out_fname)

    elif save_as in ['osm', 'pbf']:
        print('Saving graph as OSM XML. This might take a while...')
        ox.save_as_osm(
            [nodes, edges], filename=out_fname + '.osm', folder=data_dir,
            node_tags=ox.settings.osm_xml_node_tags,
            node_attrs=ox.settings.osm_xml_node_attrs,
            edge_tags=ox.settings.osm_xml_way_tags,
            edge_attrs=ox.settings.osm_xml_way_attrs,
            merge_edges=False)

        if save_as == 'pbf':
            print('Converting OSM XML to .pbf')
            os.system("osmconvert {0}.osm -o={0}.osm.pbf".format(
                os.path.join(data_dir, out_fname)))
            print('File now available at {0}'.format(
                os.path.join(data_dir, out_fname + '.osm.pbf')))
    else:
        raise ValueError(
            "{0} is not a valid output file type. See --help for more "
            "details.".format(save_as))

    odp.clean_up()
