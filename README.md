# ladot_analysis_dataprep
LADOT Analysis Tool Data Prep

This repository houses a Python script (**osm_generalized_costs.py**) designed to generate OSM-based, generalized cost-weighted networks for use in bicycle and pedestrian routing and accessibility applications. The generalized cost formulas used here are based on [Broach (2016)](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=3707&context=open_access_etds).

## How to
1. Clone/download this repository.
2. Copy local data files<sup>**</sup> into the data directory, including:
   - stop signs
   - traffic signalization
   - bikeways
   - traffic volume and speed data
3. If working with a static, local OSM extract, put your your .osm file into the data directory as well.
4. To run the analysis with all defaults, simply navigate to the `scripts/` directory of this repository and run the following command:
   ```
   python osm_slope_stats.py 
   ```
4. To use a local .osm instead of pulling OSM data from nominatim on-the-fly, you can use the `-o` flag:
   ```
   python osm_generalized_costs.py -o <your_osm_file.osm>
   ```
5. If you've already run this script before, you can save time by using the `-d` flag and pointing the script to the elevation data (DEM) .tif that was generated on-the-fly last time the script was run:
   ```
   python osm_generalized_costs.py -o <your_dem_file.tif>
   ```
7. The script will then generate an OSM XML file with the computed attributes stored as new XML tags. The following new tags are created by default:
   - gen_cost_bike:forward:left 
   - gen_cost_bike:forward:straight 
   - gen_cost_bike:forward:right 
   - gen_cost_bike:backward:left 
   - gen_cost_bike:backward:straight 
   - gen_cost_bike:backward:right 
   - gen_cost_ped:forward:left 
   - gen_cost_ped:forward:straight 
   - gen_cost_ped:forward:right 
   - gen_cost_ped:backward:left 
   - gen_cost_ped:backward:straight 
   - gen_cost_ped:backward:right
   - speed:forward:peak
   - speed:forward:offpeak
   - speed:backward:peak
   - speed:backward:offpeak
   - aadt

8. If you would rather store your output as ESRI shapefiles, simply use the `-s` flag and the script will generate two sets of shapefiles for the node and edge data, with generalized cost attributes stored in the edges. 
   ```
   python osm_generalized_costs.py -s shp
   ```

<sup>**</sup>Note: Generalized cost generation can be executed without the use of local data by running the script with the `-i` (no infrustructure data) or `-v` (no volume/speed data) flags. If your filenames are different from those specified at the top of the script, you can edit them manually there.

## Slope Computations
<img src="https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/la_mean_slopes.png" width=70%>
^ above: LA County road network colored by mean absolute slope along each OSM way.

### Examples
The following images show the LA county OSM roads colored from green to red based on the percentage of each OSM way that has a slope >= 6%:

1. This county-wide map shows roads with the highest percentage of slopes >6% clustered around the the foothills of the Santa Monica and San Gabriel mountain ranges, as expected:![](https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/la_slopes.png)

2. A more detailed view shows the severity of the slopes of streets leading down to sea level near Manhattan Beach:![](https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/manhattan_beach.png)

3. A third image highlights the slopes of roads to the north west of Dodger Stadium including the infamously inclined [Baxter Street](https://www.laweekly.com/this-super-steep-echo-park-street-is-hell-on-earth-for-cars/):  
![](https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/baxter_street.png)
 
 
 
