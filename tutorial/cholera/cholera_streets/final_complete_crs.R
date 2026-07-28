# ============================================================================
# FINAL SCRIPT: Transform Cholera Data to Transverse Mercator (OSGB 1936)
# ============================================================================
# This script transforms the cholera package street data from WGS84 
# to a custom Transverse Mercator projection based on OSGB 1936 datum
# 
# Source: WGS84 (EPSG:4326) - latitude/longitude
# Target: Transverse Mercator with OSGB 1936 datum
# 
# Parameters:
#   - Latitude of Origin: 49.0°
#   - Central Meridian: -2.0°
#   - Scale Factor: 0.9996012717
#   - False Easting: 400,000.0 m
#   - False Northing: -100,000.0 m
#   - Ellipsoid: Airy 1830
# ============================================================================

# Load required packages
if (!require("sf")) install.packages("sf")
if (!require("cholera")) install.packages("cholera")

library(sf)
library(cholera)

# ============================================================================
# STEP 1: Load and prepare the data
# ============================================================================

cat("Loading cholera street data...\n")
data(road.segments)

cat("  Rows:", nrow(road.segments), "\n")
cat("  Source coordinates: WGS84 (EPSG:4326)\n")
cat("  Location: Broad Street, Soho, London\n\n")

# ============================================================================
# STEP 2: Create line geometries from lon/lat coordinates
# ============================================================================

cat("Creating line geometries...\n")

lines <- lapply(1:nrow(road.segments), function(i) {
  st_linestring(matrix(c(road.segments$lon1[i], road.segments$lat1[i],
                         road.segments$lon2[i], road.segments$lat2[i]), 
                       ncol = 2, byrow = TRUE))
})

# Create sf object with WGS84 CRS
streets <- st_sf(
  street = road.segments$street,
  segment_id = road.segments$id,
  name = road.segments$name,
  geometry = st_sfc(lines, crs = 4326)  # WGS84
)

cat("  Created", nrow(streets), "line segments\n\n")

# ============================================================================
# STEP 3: Define target Transverse Mercator CRS with complete metadata
# ============================================================================

cat("Defining target coordinate system...\n")

# Complete WKT definition with OSGB 1936 datum
# This includes EPSG codes so GIS software will recognize it
tm_wkt <- 'PROJCS["Custom_Transverse_Mercator_OSGB1936",
    GEOGCS["GCS_OSGB_1936",
        DATUM["OSGB_1936",
            SPHEROID["Airy_1830",6377563.396,299.3249646,
                AUTHORITY["EPSG","7001"]],
            TOWGS84[446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894],
            AUTHORITY["EPSG","6277"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4277"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",49.0],
    PARAMETER["central_meridian",-2.0],
    PARAMETER["scale_factor",0.9996012717],
    PARAMETER["false_easting",400000.0],
    PARAMETER["false_northing",-100000.0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]'

# Create CRS object
tm_crs <- st_crs(tm_wkt)

cat("  Target CRS: Custom Transverse Mercator\n")
cat("  Datum: OSGB 1936 (EPSG:4277)\n")
cat("  Ellipsoid: Airy 1830 (EPSG:7001)\n\n")

# ============================================================================
# STEP 4: Transform coordinates
# ============================================================================

cat("Transforming coordinates...\n")

streets_tm <- st_transform(streets, tm_crs)

# Display extent
bbox <- st_bbox(streets_tm)
cat("  Transformed extent:\n")
cat("    Easting:  ", round(bbox["xmin"], 1), "to", round(bbox["xmax"], 1), "m\n")
cat("    Northing: ", round(bbox["ymin"], 1), "to", round(bbox["ymax"], 1), "m\n\n")

# ============================================================================
# STEP 5: Save output files
# ============================================================================

cat("Saving files...\n")

# Set output directory (change this to your preferred location)
#output_dir <- getwd()  # Current working directory
# Or specify custom directory:
output_dir <- "C:/cdm/FociInferDataAnalysis/SnowdataMap"

# Create directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save as Shapefile
shp_path <- file.path(output_dir, "cholera_streets_tm.shp")
st_write(streets_tm, shp_path, delete_dsn = TRUE, quiet = TRUE)
cat("  ✓ Shapefile:", shp_path, "\n")

# Save as GeoJSON
json_path <- file.path(output_dir, "cholera_streets_tm.geojson")
st_write(streets_tm, json_path, delete_dsn = TRUE, quiet = TRUE)
cat("  ✓ GeoJSON:", json_path, "\n")

# Save as GeoPackage
gpkg_path <- file.path(output_dir, "cholera_streets_tm.gpkg")
st_write(streets_tm, gpkg_path, delete_dsn = TRUE, quiet = TRUE)
cat("  ✓ GeoPackage:", gpkg_path, "\n")

# ============================================================================
# STEP 6: Save CRS information
# ============================================================================

info_path <- file.path(output_dir, "crs_information.txt")
sink(info_path)
cat("Cholera Streets - Coordinate Reference System Information\n")
cat("===========================================================\n\n")
cat("SOURCE CRS:\n")
cat("  Name: WGS 84\n")
cat("  EPSG: 4326\n")
cat("  Type: Geographic (latitude/longitude)\n")
cat("  Extent: -0.144° to -0.131° lon, 51.509° to 51.517° lat\n")
cat("  Location: Broad Street, Soho, London\n\n")
cat("TARGET CRS:\n")
cat("  Name: Custom Transverse Mercator based on OSGB 1936\n")
cat("  Geographic CRS: OSGB 1936 (EPSG:4277)\n")
cat("  Datum: OSGB_1936 (EPSG:6277)\n")
cat("  Ellipsoid: Airy 1830 (EPSG:7001)\n")
cat("    Semi-major axis: 6,377,563.396 m\n")
cat("    Inverse flattening: 299.3249646\n\n")
cat("PROJECTION PARAMETERS:\n")
cat("  Projection: Transverse Mercator\n")
cat("  Latitude of Origin: 49.0°\n")
cat("  Central Meridian: -2.0°\n")
cat("  Scale Factor: 0.9996012717\n")
cat("  False Easting: 400,000.0 m\n")
cat("  False Northing: -100,000.0 m\n")
cat("  Linear Unit: Metre\n\n")
cat("RESULT EXTENT:\n")
cat("  Easting:  ", bbox["xmin"], "to", bbox["xmax"], "m\n")
cat("  Northing: ", bbox["ymin"], "to", bbox["ymax"], "m\n\n")
cat("WKT DEFINITION:\n")
cat(st_crs(streets_tm)$wkt, "\n")
sink()
cat("  ✓ CRS info:", info_path, "\n")

# ============================================================================
# STEP 7: Display summary and plot
# ============================================================================

cat("\n================================================================\n")
cat("SUCCESS! Transformation complete.\n")
cat("================================================================\n")
cat("Features:", nrow(streets_tm), "street segments\n")
cat("Source: WGS84 → Target: Transverse Mercator (OSGB 1936)\n")
cat("Files saved to:", output_dir, "\n")
cat("================================================================\n\n")

# Create plot
cat("Generating plot...\n")
plot(st_geometry(streets_tm),
     main = "Cholera Streets - Transverse Mercator\nOSGB 1936 Datum",
     col = "darkblue",
     lwd = 2,
     axes = TRUE,
     xlab = "Easting (m)",
     ylab = "Northing (m)")

cat("\n✓ All done! Your data is ready for GIS analysis.\n")

# ============================================================================
# END OF SCRIPT
# ============================================================================
