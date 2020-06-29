import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import operator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_point_to_point_dists(gdf, colname='coord_pairs'):
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
    if colname not in tmp_df.columns:
        tmp_df[colname] = get_coord_pairs_from_geom(tmp_df)
    tmp_df['dists'] = tmp_df['coord_pairs'].apply(
        lambda x: np.diag(cdist(x[:-1], x[1:])))

    return tmp_df['dists']


def _get_slope_mask(gdf, lower, upper=None, direction="up"):
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


def get_coord_pairs_from_geom(gdf):
    """
    Convert geometry column to list of coordinate tuples
    Args
        gdf: A geopandas.GeoDataFrame of LineString geometries

    Returns:
        A pandas.Series containing array-like coordinate tuples
    """
    gdf['coord_pairs'] = gdf['geometry'].apply(lambda x: list(x.coords))
    return gdf['coord_pairs']


def get_slopes(gdf, z_col='z_trajectories', dist_col='dists'):
    """
    Computes point-to-point slopes along edge segments.

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
    if dist_col not in tmp_df.columns:
        tmp_df[dist_col] = _get_point_to_point_dists(tmp_df)
    tmp_df['z_diffs'] = tmp_df[z_col].apply(
        lambda x: np.diff(x))
    tmp_df['slopes'] = tmp_df['z_diffs'] / tmp_df[dist_col]

    return tmp_df['slopes']


def get_slope_stats(
        gdf, directions='all', agg_stats='mean', binned_stats=None,
        slope_pct_bins=None):
    """
    Computes statistics on input geometries for user-defined slope bins

    Args:
        gdf: A geopandas.GeoDataFrame of LineString geometries
        directions: string or list of strings defining the direction in
            which slopes should be processed. Valid values: "up", "down",
            "undirected", "all", or any list containing a combination of
            those values.
        agg_stats: string or list of strings defining the aggregate slope
            statistics to compute on each edge. Valid values: "mean",
            "min", "max", "std", "all", or a list containg any combination
            of those values.
        binned_stats: string or list of strings defining the type of
            distance statistics to compute. Valid values: "tot", "pct",
            "all", or a list containting any combination of those values.
        slope_pct_bins: list (of lists) of monotonically increasing slope
            values, e.g. `[1, 2, 3]` or `[[2, 4], [2]]`. List values are
            processed sequentially in pairs, with the two values defining
            the lower and upper bounds of slopes to be analyzed. The last
            value of any list (including single item lists) is treated as
            a lower bound with no upper bound (i.e. 3+). NOTE: bin values
            are treated as slope percentages (i.e. 1 == 1% slope).

    Returns:
        gdf with new columns for each combination of slope bin, direction,
            and statistic, e.g. "up_pct_dist_1_2", "down_tot_dist_3_plus".

    """
    if not any(isinstance(i, list) for i in slope_pct_bins):
        slope_pct_bins = [slope_pct_bins]

    gdf = gdf.copy()

    valid_directions = ["up", "down", "undirected"]
    if isinstance(directions, str):
        if directions == 'all':
            directions = ["up", "down", "undirected"]
        else:
            directions = [directions]
    if not all([x in valid_directions for x in directions]):
        raise ValueError("Unsupported slope direction specified.")

    valid_agg_stats = ['all', 'mean', 'max', 'std', 'min']
    if isinstance(agg_stats, str):
        if agg_stats == 'all':
            agg_stats = ['mean', 'max', 'std', 'min']
        else:
            agg_stats = [agg_stats]
    if not all([x in valid_agg_stats for x in agg_stats]):
        raise ValueError("Unsupported agg stat specified.")

    # generate up- and down-slope stats as well as undirected
    for direction in directions:
        logger.info("Generating slope statistics going...{0}".format(
            direction))

        for agg_stat in agg_stats:
            agg_stat_col = '{0}_{1}_slope'.format(agg_stat, direction)

            if direction == "up":
                gdf[agg_stat_col] = gdf['slopes'].apply(
                    lambda slopes: pd.Series(
                        [slope for slope in slopes if slope > 0]).agg(
                        agg_stat))

            elif direction == "down":
                gdf[agg_stat_col] = gdf['slopes'].apply(
                    lambda slopes: pd.Series(
                        [slope for slope in slopes if slope < 0]).agg(
                        agg_stat))

            elif direction == "undirected":
                gdf[agg_stat_col] = gdf['slopes'].apply(
                    lambda slopes: pd.Series(slopes).agg(agg_stat))

        if binned_stats:

            valid_binned_stats = ["tot", "pct"]
            if isinstance(binned_stats, str):
                if binned_stats == "all":
                    binned_stats = ["tot", "pct"]
                else:
                    binned_stats = [binned_stats]
            if not all([x in valid_binned_stats for x in binned_stats]):
                raise ValueError("Unsupported binned stat specified.")

            # iterate through pairs of slope boundaries defined
            for breaks in slope_pct_bins:
                for i, lower_bound in enumerate(breaks):
                    bounds = breaks[i:i + 2]

                    if len(bounds) == 2:
                        upper_bound = bounds[1]
                        upper_bound_str = str(upper_bound)

                    else:
                        upper_bound = None
                        upper_bound_str = 'plus'

                    for binned_stat in binned_stats:

                        # define the new column name to store the slope
                        # stat in the edges table
                        binned_stat_col = '{0}_{1}_dist_{2}_{3}'.format(
                            direction, binned_stat, lower_bound,
                            upper_bound_str)

                        mask = _get_slope_mask(
                            gdf, lower_bound, upper_bound, direction)

                        # multiplying the distances by the boolean mask
                        # will set all distances that correspond to slopes
                        # outside of the mask boundaries used to 0
                        masked_dists = gdf['dists'] * mask

                        # sum these masked dists to get total dist within
                        # the slope bounds
                        if binned_stat == "tot":
                            gdf[binned_stat_col] = masked_dists.apply(sum)

                        # or divide by the total edge length to get a %
                        elif binned_stat == "pct":
                            gdf[binned_stat_col] = \
                                masked_dists.apply(sum) / gdf['length']

    return gdf
