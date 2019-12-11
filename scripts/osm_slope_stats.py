import osmnx as ox
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
import os
import argparse
import operator


osm_mode = 'otf'
place = 'Los Angeles County, California, USA'
data_dir = '../data/'
osm_fname = 'la_county_hwy_only.osm'
dem_fname = 'merged.tif'
out_fname = 'la_county_slopes'
slope_stat_breaks = [2, 4, 6]
default_tags = ox.settings.useful_tags_path
addtl_tags = [
    'cycleway', 'cycleway:left', 'cycleway:right', 'bicycle', 'foot', 'access']
custom_tags = []
ox.config(useful_tags_path=default_tags + addtl_tags)


def get_point_to_point_dists(edges_df):

    tmp_df = edges_df.copy()

    # get euclidean distance between each successive
    # point of each edge's LineString geometry
    tmp_df['dists'] = tmp_df['coord_pairs'].apply(
        lambda x: np.diag(cdist(x[:-1], x[1:])))

    return tmp_df['dists']


def get_slopes(edges_df):

    tmp_df = edges_df.copy()

    # get vertical (z axis) distance between each successive
    # elevation point from each edge
    tmp_df['z_diffs'] = tmp_df['z_trajectories'].apply(
        lambda x: np.diff(x))
    tmp_df['slopes'] = tmp_df['z_diffs'] / tmp_df['dists']

    return tmp_df['slopes']


def get_slope_mask(edges_df, lower, upper=None, direction="up"):

    tmp_df = edges_df.copy()

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


if __name__ == '__main__':

    # ingest command line args
    parser = argparse.ArgumentParser(
        description='Get slope statistics for OSM network')
    parser.add_argument(
        '-o', '--osm', action='store', dest='osm',
        help='OSM XML file name')
    parser.add_argument(
        '-d', '--data-dir', action='store', dest='data_dir',
        help='path to data directory')
    parser.add_argument(
        '-m', '--osm-mode', action='store', dest='mode',
        help='"local" or "osm"')

    options = parser.parse_args()

    if options.osm:
        osm_mode = 'local'
        osm_fname = options.osm
    if options.mode:
        osm_mode = options.mode
    if options.data_dir:
        data_dir = options.data_dir

    # load local osm data
    print('Loading OSM data...')
    if osm_mode == 'local':
        path = os.path.join(data_dir, osm_fname)
        try:
            G = ox.graph_from_file(path, simplify=False, retain_all=True)
        except OSError:
            print(
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
    G = ox.simplify_graph(G, strict=False)

    # Remove edges that OSMnx duplicated when converting
    # OSM XML to a NetworkX MultiDiGraph
    H = ox.get_undirected(G)

    # extract nodes/edges geodataframes and project them
    # into equidistant, meters-based coordinate system
    nodes, edges = ox.graph_to_gdfs(H)
    edge_cols = edges.columns
    nodes.crs = {'init': 'epsg:4326'}
    edges.crs = {'init': 'epsg:4326'}
    edges = edges.to_crs(epsg=2770)

    # process the geometries to perform calculations
    edges['coord_pairs'] = edges['geometry'].apply(lambda x: list(x.coords))
    print('Done.')

    # load elevation data
    print('Loading the DEM.')
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
        for i, lower_bound in enumerate(slope_stat_breaks):
            bounds = slope_stat_breaks[i:i + 2]

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

    # project the edges back to lat/lon coordinate system
    edges = edges.to_crs(epsg=4326)

    # turn the edges back to a graph
    print('Converting edges and nodes back to graph structure.')
    G = ox.gdfs_to_graph(nodes, edges)
    print('Done.')

    # save the graph back to disk as shapefile data
    path = os.path.join(data_dir, out_fname)
    print('Saving the data back to disk at {0}'.format(path))
    ox.save_graph_shapefile(G, out_fname, data_dir)
