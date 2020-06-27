import numpy as np
from scipy.spatial.distance import cdist
import requests
import os
import operator
import zipfile
from osgeo import gdal
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from tqdm import tqdm
import shutil

dem_formattable_url = (
    'https://prd-tnm.s3.amazonaws.com/StagedProducts/'
    'Elevation/1/TIFF/n{0}w{1}/USGS_1_n{0}w{1}.tif')


class OpenDEMProfiler(object):
    """Generates high resolution elevation profiles on the fly

    This collection of methods is designed to generate elevation profiles
    and slope statistics for linear feature geometries (e.g. LineStrings)
    using open source USGS DEMs

    Attributes:
        dem_formattable_url (str): A generic formattable string defining the
            URL path to the DEM data on the USGS server
        data_dir (str): The path to the local data directory
        local_crs (str): Defines the local coordinate reference system to
            use. Should be a crs where units are meters.
    """

    def __init__(
            self,
            dem_formattable_url=dem_formattable_url,
            data_dir='./data/', local_crs='EPSG:2770'):

        self.dem_formattable_url = dem_formattable_url
        self.data_dir = data_dir
        self.local_crs = local_crs

    def _get_coord_pairs_from_geom(self, gdf):
        """
        Convert geometry column to list of coordinate tuples
        Args
            gdf: A geopandas.GeoDataFrame of LineString geometries

        Returns:
            A pandas.Series containing array-like coordinate tuples
        """
        gdf['coord_pairs'] = gdf['geometry'].apply(lambda x: list(x.coords))
        return gdf['coord_pairs']

    def _format_dem_url(self, x, y):
        """
        Construct full url of USGS DEM file.

        Args:
            x: 2-digit longitude
            y: 3-digit latitude

        Returns:
            full url of USGS DEM file.
        """
        full_url = self.dem_formattable_url.format(y, x)
        return full_url

    def _download_save_geotiff(self, url, fname):
        """
        Download USGS GeoTIFF file from url

        Args:
            url: full url of USGS GeoTIFF file
            fname: name of geotiff file on disk
        """
        res = requests.get(url)
        directory = os.path.join(self.data_dir, 'tmp')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, fname + '.tif'), 'wb') as f:
            f.write(res.content)
        return

    def _download_save_unzip_dem(self, url, fname):
        """
        Downloads zipped archive of DEM data from USGS and extracts
        all files to disk

        Args:
            url: full url of the zipped USGS dem data
            fname: name of geotiff file on disk
        """
        res = requests.get(url)
        zipped_fname = os.path.join(self.data_dir, fname + '.zip')

        with open(zipped_fname, 'wb') as foo:
            foo.write(res.content)

        directory = os.path.join(self.data_dir, fname)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with zipfile.ZipFile(zipped_fname, 'r') as zip_ref:
            zip_ref.extractall(directory)

        return

    def _convert_adf_to_gtiff(self, fname):
        """
        Convert ArcGIS binary grid file raster data to geotiff

        Args:
            fname: name of geotiff file on disk

        """

        in_fname = glob.glob(os.path.join(
            self.data_dir, fname, '**', 'w001001.adf'))[0]
        src_ds = gdal.Open(in_fname)
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.CreateCopy(self.data_dir + fname + '.tif', src_ds, 0)
        del dst_ds
        del src_ds

        return

    def _get_all_dems(self, integer_bbox):
        """
        Download all 1-arc second DEM data needed to cover
        OSM network bounds

        Args:
            integer_bbox: A tuple of min_lon, min_lat,
                max_lon, max_lat integer coordinates

        Returns:
            The number of total files downloaded.
        """
        min_x, min_y, max_x, max_y = integer_bbox
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
                url = self._format_dem_url(x, y)
                _ = self._download_save_geotiff(url, fname)
                it += 1
                print(
                    'Downloaded {0} of {1} DEMs and saved as GeoTIFF.'.format(
                        it, tot_files))

        return tot_files

    def _get_mosaic(self, fname):
        """
        Combine individual GeoTIFFs into a single file

        Args:
            fname: name of geotiff file on disk
        """
        tmp_dir = os.path.join(self.data_dir, 'tmp')
        all_tif_files = glob.glob(os.path.join(tmp_dir, '*.tif'))
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
        mosaic_path = os.path.join(tmp_dir, fname)
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(merged)

        return

    def _reproject_geotiff(self, fname):
        """
        Takes a geotiff, reprojects it and saves new file to disk

        Args:
            fname: name of geotiff file on disk

        """
        tmp_data_dir = os.path.join(self.data_dir, 'tmp', fname)
        geotiff_output_path = os.path.join(self.data_dir, fname)

        with rasterio.open(tmp_data_dir) as src:
            dst_crs = self.local_crs
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(geotiff_output_path, 'w', **kwargs) as dst:
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

    def download_usgs_dem(self, integer_bbox, dem_fname):
        """
        Downloads DEM files from USGS API

        Arguments:
            integer_bbox (tuple): A tuple of min_lon, min_lat,
                max_lon, max_lat integer coordinates

        Returns:
            None
        """

        num_files = self._get_all_dems(integer_bbox)

        if num_files > 1:
            _ = self._get_mosaic(dem_fname)

        else:
            single_file = glob.glob(
                os.path.join(self.data_dir, 'tmp', '*.tif'))[0]
            shutil.copyfile(
                single_file, os.path.join(self.data_dir, 'tmp', dem_fname))
        _ = self._reproject_geotiff(dem_fname)

        return

    def get_point_to_point_dists(self, gdf, colname='coord_pairs'):
        """
        Fetches pairwise euclidean distances from a dataframe
        of network edges containing lists of (x,y) coordinate pairs,
        each of which corresponds to a point in the LineString
        geometry of an edge.

        Arguments:
            gdf: A geopandas.GeoDataFrame of LineString geometries
            colname: The name of the column in gdf containing the
                coordinate pair tuples

        Returns:
            A pandas.Series object with lists of distances as its
                values.
        """
        tmp_df = gdf.copy()
        tmp_df['dists'] = tmp_df['coord_pairs'].apply(
            lambda x: np.diag(cdist(x[:-1], x[1:])))

        return tmp_df['dists']

    def get_z_trajectories(self, gdf, dem):
        """
        Generate elevation trajectories for each coordinate pair

        Args
            gdf: A geopandas.GeoDataFrame of LineString geometries
            dem: A rasterio.DatasetReader object, probably a
                Digital Elevation Model (DEM)

        Returns:
            A pandas.Series containing array-like elevation profiles
        """
        z_trajectories = []
        for i, edge in tqdm(gdf.iterrows(), total=len(gdf)):
            z_trajectories.append(
                [x[0] for x in dem.sample(edge['coord_pairs'])])

        return gdf['z_trajectories']

    def get_slopes(self, gdf, z_col='z_trajectories', dist_col='dists'):
        """
        Computes slopes along edge segments.

        Using vertical (z-axis) trajectories and lists of edge
        segment distances, calculates the slope along each segment
        of a LineString geometry for every edge.

        Arguments:
            gdf: A geopandas.GeoDataFrame of LineString geometries
            z_col: The name of the column in gdf containing elevation
                trajectories for each feature
            dist_col: The name of the column in gdf containing point-to-
                point distance tuples for each feature


        Returns:
            A pandas.Series object with lists of slopes as its values.
        """
        tmp_df = gdf.copy()
        tmp_df['z_diffs'] = tmp_df[z_col].apply(
            lambda x: np.diff(x))
        tmp_df['slopes'] = tmp_df['z_diffs'] / tmp_df[dist_col]

        return tmp_df['slopes']

    def get_slope_mask(self, gdf, lower, upper=None, direction="up"):
        """
        Generates an array of booleans that can be used to mask
        other arrays based on their position relative to user-defined
        slope boundaries.

        Args:
            gdf: A geopandas.GeoDataFrame of LineString geometries
            lower: a numeric lower bound to use for filtering slopes
            upper: a numeric upper bound to use for filtering slopes
            direction: one of ["up", "down", "undirected"]

        Returns:
            A pandas.Series of boolean values
        """
        tmp_df = gdf.copy()

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

    def clean_up(self):
        """
        clear out the tmp directory
        """
        tmp_dir = os.path.join(self.data_dir, 'tmp')
        tmp_files = os.path.join(tmp_dir, "*")
        for f in glob.glob(tmp_files):
            os.remove(f)
