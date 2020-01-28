# ladot_analysis_dataprep
LADOT Analysis Tool Data Prep

## Slope Computations
<img src="https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/la_mean_slopes.png" width=70%>
^ above: LA County road network colored by mean absolute slope along each OSM way.

### How to
1. Put .tif DEM data into the data directory of this repository
2. If working with local OSM data, put your local .osm file into the data directory as well.
3. To run the analysis with all defaults, simply navigate to the `scripts/` directory of this repository and run the following command:
```
python osm_slope_stats.py 
```
4. To use a local .osm instead of pulling OSM data from nominatim on-the-fly, you have two options
   - `python osm_slope_stats.py -m local` to use the local file defined in the header of the script
   - `python osm_slope_stats.py -o <file.osm>` to specify the name of the local file to use at runtime.
5. The script will then generate ESRI shapefiles of OSM node and edge data with the slope statistics stored in the edges

### Examples
The following images show the LA county OSM roads colored from green to red based on the percentage of each OSM way that has a slope >= 6%:

1. This county-wide map shows roads with the highest percentage of slopes >6% clustered around the the foothills of the Santa Monica and San Gabriel mountain ranges, as expected:![](https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/la_slopes.png)

2. A more detailed view shows the severity of the slopes of streets leading down to sea level near Manhattan Beach:![](https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/manhattan_beach.png)

3. A third image highlights the slopes of roads to the north west of Dodger Stadium including the infamously inclined [Baxter Street](https://www.laweekly.com/this-super-steep-echo-park-street-is-hell-on-earth-for-cars/):  
![](https://github.com/RSGInc/ladot_analysis_dataprep/blob/master/baxter_street.png)
 
 
 
