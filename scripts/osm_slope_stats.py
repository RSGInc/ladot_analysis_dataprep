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
from glob import glob
import urllib.request as request
import shutil
from contextlib import closing


osm_mode = 'otf'
dem_mode = 'otf'
# place = 'Los Angeles County, California, USA'
place = 'San Francisco, California, United States'
place_for_fname_str = place.split(',')[0].replace(' ', '_')
data_dir = '../data/'
osm_fname = 'la_county_hwy_only.osm'
dem_fname = '{0}.tif'.format(place_for_fname_str)
out_fname = '{0}_slopes'.format(place_for_fname_str)
dem_formattable_path = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/n{0}w{1}/'
dem_formattable_fname = 'USGS_1_n{0}w{1}.tif'
slope_stat_breaks = [2, 4, 6]
default_tags = ox.settings.useful_tags_path
addtl_tags = [
    'cycleway', 'cycleway:left', 'cycleway:right', 'bicycle', 'foot', 'access']
custom_tags = []
ox.config(useful_tags_path=default_tags + addtl_tags)


def get_integer_bbox(nodes_df):
    min_x = int(np.abs(np.floor(nodes_df['x'].min())))
    min_y = int(np.abs(np.ceil(nodes_df['y'].min())))
    max_x = int(np.abs(np.floor(nodes_df['x'].max())))
    max_y = int(np.abs(np.ceil(nodes_df['y'].max())))
    return min_x, min_y, max_x, max_y


def format_dem_url(
        x, y, dem_formattable_path=dem_formattable_path,
        dem_formattable_fname=dem_formattable_fname):
    formatted_path = dem_formattable_path.format(y, x)
    formatted_fname = dem_formattable_fname.format(y, x)
    full_url = formatted_path + formatted_fname
    return full_url


def download_save_geotiff(url, fname, data_dir=data_dir):
    res = requests.get(url)
    directory = os.path.join(data_dir, 'tmp')
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, fname + '.tif'), 'wb') as f:
        f.write(res.content)
    return


def download_save_unzip_dem(url, fname, data_dir=data_dir):
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


def convert_adf_to_gtiff(fname, data_dir):

    in_fname = glob(os.path.join(data_dir, fname, '**', 'w001001.adf'))[0]
    src_ds = gdal.Open(in_fname)
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.CreateCopy(data_dir + fname + '.tif', src_ds, 0)
    dst_ds = None
    src_ds = None

    return


def get_all_dems(
        min_x, min_y, max_x, max_y, dem_formattable_path,
        dem_formattable_fname=dem_formattable_fname):
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


def get_mosaic(data_dir, out_fname=dem_fname):
    all_tif_files = glob(os.path.join(data_dir, '*.tif'))
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
            os.path.join(data_dir, 'tmp', out_fname), "w", **out_meta) as dest:
        dest.write(merged)


def reproject_geotiff(fname, data_dir):
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


def get_point_to_point_dists(edges_df):
    """
    Fetches pairwise euclidean distances from a dataframe
    of network edges containing lists of (x,y) coordinate pairs,
    each of which corresponds to a point in the LineString
    geometry of an edge.

    Args:
        edges_df: a pandas.DataFrame object with a column
            named 'coord_pairs' containing a list of
            consecutive (x,y) coordinate pairs.

    Returns:
        A pandas.Series object with lists of distances as its
            values.
    """

    tmp_df = edges_df.copy()
    tmp_df['dists'] = tmp_df['coord_pairs'].apply(
        lambda x: np.diag(cdist(x[:-1], x[1:])))

    return tmp_df['dists']


def get_slopes(edges_df):
    """
    Computes slopes along edge segments.

    Using vertical (z-axis) trajectories and lists of edge
    segment distances, calculates the slope along each segment
    of a LineString geometry for every edge.

    Args:
        edges_df: a pandas.DataFrame object with columns
            named 'z_trajectories' and 'z_dists'.

    Returns:
        A pandas.Series object with lists of slopes as its values.
    """

    tmp_df = edges_df.copy()
    tmp_df['z_diffs'] = tmp_df['z_trajectories'].apply(
        lambda x: np.diff(x))
    tmp_df['slopes'] = tmp_df['z_diffs'] / tmp_df['dists']

    return tmp_df['slopes']


def get_slope_mask(edges_df, lower, upper=None, direction="up"):
    """
    Generates an array of booleans that can be used to mask
    other arrays based on their position relative to user-defined
    boundaries.

    Args:
        edges_df: a pandas.DataFrame object with a column
            named 'slopes' containing a list of edge segment
            slopes
        lower: a numeric lower bound to use for filtering slopes
        upper: a numeric upper bound to use for filtering slopes
        direction: one of ["up", "down", "undirected"]

    Returns:
        A pandas.Series of boolean values
    """

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
        '-m', '--osm-mode', action='store', dest='osm_mode',
        help='"local" or "otf"')
    parser.add_argument(
        '-d', '--dem-mode', action='store', dest='dem_mode',
        help='"local" or "otf"')
    parser.add_argument(
        '-p', '--place', action='store', dest='place',
        help='valid nominatim place name')

    options = parser.parse_args()

    if options.osm:
        osm_mode = 'local'
        osm_fname = options.osm
    if options.osm_mode:
        osm_mode = options.osm_mode
    if options.dem_mode:
        dem_mode = options.dem_mode
    if options.place:
        place = options.place
        place_for_fname_str = place.split(',')[0].replace(' ', '_')
        dem_fname = '{0}.tif'.format(place_for_fname_str)
        out_fname = '{0}_slopes'.format(place_for_fname_str)

    print('Let get slope statistics for {0} roads!'.format(place))

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
    if dem_mode == 'otf':
        print('Downloading DEMs from USGS...this might take a while')
        integer_bbox = get_integer_bbox(nodes)
        num_files = get_all_dems(*integer_bbox, dem_formattable_path, dem_formattable_fname)

        if num_files > 1:
            _ = get_mosaic(data_dir, dem_fname)
        else:
            single_file = glob(os.path.join(data_dir, 'tmp', '*.tif'))[0]
            shutil.copyfile(single_file, os.path.join(data_dir, 'tmp', dem_fname))
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
    print('Saving the data to disk at {0}'.format(path))
    ox.save_graph_shapefile(G, out_fname, data_dir)

    # clear out the tmp directory
    tmp_files = glob('../data/tmp/*')
    for f in tmp_files:
        os.remove(f)
