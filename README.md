# LADOT Analysis Tool Data Prep

This repository houses Python scripts to build networks and land use data for accessibility applications.

# Network
The **osm_generalized_costs.py** script is designed to generate OSM-based, generalized cost-weighted networks for bicycle and pedestrian accessibility. The generalized cost formulas used here are an adaptation of [Broach (2016)](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=3707&context=open_access_etds).  

## How to Build the Network
1. Clone/download this repository.
2. Copy local data files<sup>&dagger;</sup> into the data directory, including:
   - stop signs
   - traffic signalization
   - bikeways
   - crosswalks
   - traffic volume and speed data
3. If working with a static, local OSM extract, put your your .osm file into the data directory as well.
4. To run the analysis with all defaults, simply navigate to the `scripts/` directory of this repository and run the following command:
   ```
   python osm_generalized_costs.py 
   ```
4. To use a local .osm instead of pulling OSM data from nominatim on-the-fly, you can use the `-o` flag:
   ```
   python osm_generalized_costs.py -o <your_osm_file.osm>
   ```
5. If you've already run this script before, you can save time by using the `-d` flag and pointing the script to the elevation data (DEM) .tif that was generated on-the-fly last time the script was run:
   ```
   python osm_generalized_costs.py -o <your_dem_file.tif>
   ```
7. The script will then generate an OSM XML file with the computed attributes stored as new OSM way tags. The following new tags are created by default:
   - `gen_cost_bike:forward:link`
   - `gen_cost_bike:forward:left`
   - `gen_cost_bike:forward:straight`
   - `gen_cost_bike:forward:right`
   - `gen_cost_bike:backward:link`
   - `gen_cost_bike:backward:left`
   - `gen_cost_bike:backward:straight`
   - `gen_cost_bike:backward:right`
   - `gen_cost_ped:forward:link`
   - `gen_cost_ped:forward:left`
   - `gen_cost_ped:forward:straight`
   - `gen_cost_ped:forward:right`
   - `gen_cost_ped:backward:link`
   - `gen_cost_ped:backward:left`
   - `gen_cost_ped:backward:straight`
   - `gen_cost_ped:backward:right`
   - `speed_peak:forward`
   - `speed_offpeak:forward`
   - `speed_peak:backward`
   - `speed_offpeak:backward`
   - `aadt`

8. If you would rather store your output as ESRI shapefiles, simply use the `-s` flag and the script will generate two sets of shapefiles for the node and edge data, with generalized cost attributes stored in the edges. 
   ```
   python osm_generalized_costs.py -s shp
   ```

<sup>&dagger;</sup>Note: Generalized cost generation can be executed without the use of local data by running the script with the `-i` (no infrustructure data) or `-v` (no volume/speed data) flags. If you do want to use local data but your filenames are different from those specified at the top of the script, you can edit them manually there.

## Generalized Costs Calculations

### Bicycle
| Length Adjusted Metric    | Length Multiplier<sup>*</sup> | Variable Name            | Notes                                                                                      |
|---------------------------|-------------------------------|--------------------------|--------------------------------------------------------------------------------------------|
| distance                  | 1.0                           | distance                 |                                                                                            |
| bike boulevard            | -0.108                        | bike_blvd_penalty        | OSM: cycleway="shared" OR LADOT: bikeway=("Route" OR "Shared Route")                       |
| bike path                 | -0.16                         | bike_path_penalty        | OSM: highway="cycleway" OR (highway="path" & bicycle="dedicated") OR LADOT: bikeway="Path" |
| prop link slope 2-4%      | 0.371                         | slope_penalty            | upslope for forward direction, downslope for backward direction                            |
| prop link slope 4-6%      | 1.23                          | slope_penalty            | upslope for forward direction, downslope for backward direction                            |
| prop link slope 6%+       | 3.239                         | slope_penalty            | upslope for forward direction, downslope for backward direction                            |
| no bike lane (10-20k)     | 0.368                         | no_bike_penalty          | OSM: cycleway=(NULL OR "no") OR OSM: bicycle="no" AND LADOT: bikeway=NULL                  |
| no bike lane (20-30k)     | 1.4                           | no_bike_penalty          | OSM: cycleway=(NULL OR "no") OR OSM: bicycle="no" AND LADOT: bikeway=NULL                  |
| no bike lane (30k+)       | 7.157                         | no_bike_penalty          | OSM: cycleway=(NULL OR "no") OR OSM: bicycle="no" AND LADOT: bikeway=NULL                  |

| Fixed Distance Metric  | Addt'l Distance (m)<sup>*</sup> | Variable Name            | Notes                                                                  |
|------------------------|---------------------------------|--------------------------|------------------------------------------------------------------------|
| turns                  | 54                              | turn_penalty             | assume additive ped turn penalty and scale other penalties accordingly |
| stop signs             | 6                               | stop_penalty             | (LADOT: stop/yield)                                                    |
| traffic signal         | 27                              | signal_penalty           | (LADOT: signalized intersection)                                       |
| cross traffic (5-10k)  | 78                              | cross_traffic_penalty_ls | left or straight only                                                  |
| cross traffic (10-20k) | 81                              | cross_traffic_penalty_ls | left or straight only                                                  |
| cross traffic (20k+)   | 424                             | cross_traffic_penalty_ls | left or straight only                                                  |
| cross traffic (10k+)   | 50                              | cross_traffic_penalty_r  | right only                                                             |
| parallel traffic (10-20k) | ?                        | parallel_traffic_penalty |                                                                                            |
| parallel traffic (20k+)   | ?                         | parallel_traffic_penalty |                                                                                            |

<sup>*</sup>Multipliers and distances inspired by Broach (2016)

| Generalized Cost       | Formula                                                                                                                    |
|------------------------|----------------------------------------------------------------------------------------------------------------------------|
| gen_cost_bike:link     | distance + distance * (bike_blvd_penalty + bike_path_penalty + slope_penalty + no_bike_penalty) |
| gen_cost_bike:left     | turn_penalty + stop_penalty + signal_penalty + cross_traffic_penalty_ls + parallel_traffic_penalty|
| gen_cost_bike:straight | stop_penalty + signal_penalty + cross_traffic_penalty_ls|
| gen_cost_bike:right    | turn_penalty + stop_penalty + signal_penalty + cross_traffic_penalty_r |

#### Examples:
| | South Budlong Ave. | Baxster Street | 
|--|--|--|
|Way ID |[165344383](https://www.openstreetmap.org/way/165344383#map=19/33.97519/-118.29605)| [161705335](https://www.openstreetmap.org/way/161705335)
| From Node | 123058787 | 5531221585 |
| To Node | 123058790 | 26187155 |
| Length | 89.023 | 225.923 |
| gen_cost_bike:forward:left | 94.631449 | 828.0568817 |
| gen_cost_bike:forward:straight | 90.892483 | 819.9501257 |
| gen_cost_bike:forward:right | 92.761966 | 828.0568817 |
| gen_cost_bike:backward:left | 93.207081 | 202.089846 |
| gen_cost_bike:backward:straight | 89.468115 | 193.98309 |
| gen_cost_bike:backward:right | 93.207081 | 202.089846 |
| slope_penalty:forward | 0 | 3.243050056 |
| slope_penalty:backward | 0 | 0 |
| bike_path_penalty:forward | 0 | 0 |
| bike_path_penalty:backward | 0 | 0 |
| bike_blvd_penalty:forward | 0 | 0 |
| bike_blvd_penalty:backward | 0 | 0 |
| signal_penalty:forward | 0.021 | 0 |
| signal_penalty:backward | 0 | 0 |
| stop_sign_penalty:forward | 0 | 0.005 |
| stop_sign_penalty:backward | 0.005 | 0.005 |

### Pedestrian

| Length Adjusted Metric  |   Length Multiplier<sup>*</sup> | Variable Name         | Notes                                                                                                                                                                     |
|-------------------------|---------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| distance                | 1.0                             | distance              |                                                                                                                                                                           |
| prop link slope 10%+    | 0.99                            | ped_slope_penalty     | upslope for forward direction, downslope for backward direction                                                                                                           |
| unpaved or alleyway     | 0.51                            | unpaved_alley_penalty | OSM: highway="alley" OR surface="unpaved"                                                                                                                                 |
| busy street             | 0.14                            | busy_penalty          | OSM: highway=('tertiary' OR 'tertiary_link' OR 'secondary' OR 'secondary_link' OR 'primary' OR 'primary_link' OR 'trunk' OR 'trunk_link' OR 'motorway' OR 'motorway_link' |
| neighborhood commercial | -0.28                           | nbd_penalty           |                                                                                                                                                                           |

| Fixed Distance Metric            | Addt'l Distance (m)<sup>*</sup> | Variable Name              | Notes                                                                                                                                                                                   |
|----------------------------------|---------------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| turn                             | 54                              | turn_penalty               |                                                                                                                                                                                         |
| unsignalized arterial crossing   | 73                              | unsig_art_xing_penalty     | left or right: ((13000 <= parallel traffic AADT <= 23000) OR (13000 <= self-edge AADT <= 23000)) AND (unsignalized) straight: (13000 <= cross traffic AADT <= 23000) AND (unsignalized) |
| collector crossing w/o crosswalk | 28                              | unmarked_coll_xing_penalty | left or right: ((10000 <= parallel traffic AADT < 13000) OR (10000 <= self-edge AADT < 13000)) AND (no crosswalk) straight: (10000 <= cross traffic AADT < 13000) AND (no crosswalk)    |

<sup>*</sup>Multipliers and distances inspired by Broach (2016)

| Generalized Cost      | Formula                                                                                    |
|-----------------------|--------------------------------------------------------------------------------------------|
| gen_cost_ped:link     | distance + distance * (slope_penalty + unpaved_alley_penalty + busy_penalty + nbd_penalty) |
| gen_cost_ped:left     | turn_penalty + unsig_art_xing_penalty + unmarked_coll_xing                                 |
| gen_cost_ped:straight | turn_penalty + unsig_art_xing_penalty + unmarked_coll_xing                                 |
| gen_cost_ped:right    | turn_penalty + unsig_art_xing_penalty + unmarked_coll_xing                                 |

#### Examples:
| | Lanark Street|
|--|--|
|Way ID | [13356087](https://www.openstreetmap.org/way/13356087)|
|From Node | 123018756 |
|To Node | 368008589 |
|gen_cost_ped:forward:left | 108.416 |
|gen_cost_ped:forward:straight | 54.416 |
|gen_cost_ped:forward:right | 108.416 |
|gen_cost_ped:backward:left | 108.416 |
|gen_cost_ped:backward:straight | 127.416 |
|gen_cost_ped:backward:right | 108.416 |
|unsig_art_xing_penalty_lr:forward | 0 |
|unsig_art_xing_penalty_s:forward | 0 |
|unsig_art_xing_penalty_lr:backward | 0 |
|unsig_art_xing_penalty_s:backward | 73 |
|unmarked_coll_xing_penalty_lr:forward | 0 |
|unmarked_coll_xing_penalty_s:forward | 0 |
|unmarked_coll_xing_penalty_lr:backward | 0 |
|unmarked_coll_xing_penalty_s:backward | 0 |
|ped_slope_penalty:forward | 0 |
|ped_slope_penalty:backward | 0 |
|unpaved_alley_penalty | 0 |
|busy_penalty | 0 |

## Control Type Assignment

### Stop Signs
Currently stop sign designations are assigned at the intersection level, meaning if there is any stop sign at an intersection, all edges terminating at that intersection are assigned a stop sign penalty:
<img src="images/stop_sign_matching.png" width=70%>

## Pedestrian Infrastructure Assignment

### Crosswalks
Crosswalk assignment currently works like stop sign assignment described above. If there is a crosswalk at an intersection, all edges terminating at that intersection are assigned a crosswalk penalty:

<img src="images/xwalk_matching.png">
If OSM has footway edges representing the crosswalks, then those footways will be associated with the crosswalk, as seen in the right-most intersection above. Otherwise, the crosswalks will be associated with the roadway edges as seen in the two intersections to the left.

## Bicycle Infrastructure Assignment

### Bike Lanes
Bike infrastructure is assigned by converting LADOT Bikeways lines to points, and then snapping those points to the OSM network:

<img src="images/bike_infra_matching.png" width=70%>
^ Above, LADOT Bikeways are shown in teal, with OSM ways shown in pink where they have been assigned bicycle infrastructure and blue where they have not.

## Slope Computations
<img src="images/la_mean_slopes.png" width=70%>
^ above: LA County road network colored by mean absolute slope along each OSM way.

### Examples
The following images show the LA county OSM roads colored from green to red based on the percentage of each OSM way that has a slope >= 6%:

1. This county-wide map shows roads with the highest percentage of slopes >6% clustered around the the foothills of the Santa Monica and San Gabriel mountain ranges, as expected:<img src="images/la_slopes.png"  width=70%>

2. A more detailed view shows the severity of the slopes of streets leading down to sea level near Manhattan Beach: <img src="images/manhattan_beach.png"  width=70%>

3. A third image highlights the slopes of roads to the NW of Dodger Stadium, including the infamously inclined [Baxter Street](https://www.laweekly.com/this-super-steep-echo-park-street-is-hell-on-earth-for-cars/):
   
   <img src="images/baxter_street.png"  width=70%>
 
# Land Use Data
The following datasets are used by Conveyal to define "opportunities" for computing accessibilities and are not required for computing generalized costs on the travel network:
   - **Land Use** - Additional land use data for use in Conveyal Analysis are available as shapefiles [here](https://resourcesystemsgroupinc-my.sharepoint.com/personal/ben_stabler_rsginc_com1/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9yZXNvdXJjZXN5c3RlbXNncm91cGluYy1teS5zaGFyZXBvaW50LmNvbS86ZjovZy9wZXJzb25hbC9iZW5fc3RhYmxlcl9yc2dpbmNfY29tMS9FZ205c3hEa2V0NU5tNHBoTEZ2X05zNEJJSjlNZlY1anN6NFM4SHFUNnc2c0ZnP3J0aW1lPTRLZjEyMHpCMTBn&CT=1584409358939&OR=OWA%2DNT&CID=6d731269%2D125c%2Dd14b%2Db1fe%2De357a419fd64&id=%2Fpersonal%2Fben%5Fstabler%5Frsginc%5Fcom1%2FDocuments%2FLADOT%2FLand%20Use). Their contents have been documented in the LADOT_landuse_data_inventory.xlsx available [here](https://resourcesystemsgroupinc-my.sharepoint.com/:x:/g/personal/ben_stabler_rsginc_com1/Ec1iip5AnGlJjmogrsbKyrYBkBqCd9hhzevPk-j-_ox57w?e=M4tZH3).
   - **Census** - The script to generate Census-based population and household data stored as shapefiles is located in the `scripts/` directory of this repository. The latest data as of March 2020 is included in the Land Use data files linked above.
