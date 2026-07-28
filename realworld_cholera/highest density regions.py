import arcpy
import numpy as np

# 1. Convert your Raster to a NumPy Array
raster_path = "KernelD_sin13"  # The name of your layer in ArcGIS
array = arcpy.RasterToNumPyArray(raster_path, nodata_to_value=0)

# 2. Flatten the array to a simple list of values
flat_array = array.flatten()

# 3. Sort values from High (Peak) to Low (Valley)
sorted_values = np.sort(flat_array)[::-1]

# 4. Calculate Cumulative Sum
# (This sums up the "mass" of the probability as we go down the mountain)
cumsum = np.cumsum(sorted_values)
total_sum = cumsum[-1]

# 5. Find the threshold index where we hit 95% of the total mass
# Change 0.95 to 0.50 for the 50% CI (the "core")
target_volume = 0.95 * total_sum
index_cutoff = np.searchsorted(cumsum, target_volume)

# 6. Get the Pixel Value at that index
contour_level_95 = sorted_values[index_cutoff]

print(f"Set your ArcGIS Contour tool to this value: {contour_level_95}")



import arcpy
import numpy as np

# 1. Convert your Raster to a NumPy Array
raster_path = "KernelD_sin13"  # The name of your layer in ArcGIS
array = arcpy.RasterToNumPyArray(raster_path, nodata_to_value=0)

# 2. Flatten the array to a simple list of values
flat_array = array.flatten()

# 3. Sort values from High (Peak) to Low (Valley)
sorted_values = np.sort(flat_array)[::-1]

# 4. Calculate Cumulative Sum
# (This sums up the "mass" of the probability as we go down the mountain)
cumsum = np.cumsum(sorted_values)
total_sum = cumsum[-1]

# 5. Find the threshold index where we hit 95% of the total mass
# Change 0.95 to 0.50 for the 50% CI (the "core")
target_volume = 0.50 * total_sum
index_cutoff = np.searchsorted(cumsum, target_volume)

# 6. Get the Pixel Value at that index
contour_level_50 = sorted_values[index_cutoff]

print(f"Set your ArcGIS Contour tool to this value: {contour_level_50}")


