# MRI Preprocessing Script for Pituitary Region 
# Version: Final (07.11.2024)

import SimpleITK as sitk
import numpy as np
import pandas as pd
import neuroCombat
import os
from time import time

# Define paths at the beginning for better maintainability
PATHS = {
    'input_image': r"<path_to_input_image>.nrrd",
    'sellar_mask': r"<path_to_sellar_mask>.nrrd",
    'output_image': r"<path_to_output_image>.nrrd"
}

# Record start time
start_time = time()

# Step 1: Load the Original Image
try:
    image = sitk.ReadImage(PATHS['input_image'], sitk.sitkFloat32)
    print("Image loaded successfully.")
    print("Image size:", image.GetSize())
except Exception as e:
    print(f"Error loading image: {e}")
    raise

# Step 2: Create a Mask for the Region of Interest (Head Mask)
try:
    transformed = sitk.RescaleIntensity(image, 0, 255)
    head_mask = sitk.LiThreshold(transformed, 0, 1)
    print("Head mask created successfully.")
except Exception as e:
    print(f"Error creating mask: {e}")
    raise

# Step 3: Shrink the Image and Mask for Faster Processing
shrink_factor = 2
try:
    print("Applying shrink factor...")
    input_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    mask_image = sitk.Shrink(head_mask, [shrink_factor] * head_mask.GetDimension())
    print("Shrink applied successfully.")
except Exception as e:
    print(f"Error applying shrink: {e}")
    raise

# Step 4: Apply Bias Field Correction
try:
    print("Applying bias field correction... Please wait.")
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50])
    corrector.SetNumberOfControlPoints([4, 4, 4])
    corrected_image_shrinked = corrector.Execute(input_image, mask_image)
    print("Bias field correction applied successfully on the shrinked image.")
except Exception as e:
    print(f"Error during bias field correction: {e}")
    raise

# Step 5: Restore Corrected Image to Original Resolution
try:
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
    print("Corrected image restored to full resolution successfully.")
except Exception as e:
    print(f"Error restoring corrected image to full resolution: {e}")
    raise

# Step 6: Apply Z-Score Normalization to the Corrected Image
try:
    sellar_mask = sitk.ReadImage(PATHS['sellar_mask'], sitk.sitkUInt8)
    print("Sellar region mask loaded successfully.")
    
    masked_image = sitk.Mask(corrected_image_full_resolution, sellar_mask)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(masked_image)
    mean_intensity = stats.GetMean()
    std_intensity = stats.GetSigma()

    corrected_np = sitk.GetArrayFromImage(corrected_image_full_resolution)
    corrected_np_normalized = (corrected_np - mean_intensity) / std_intensity

    corrected_image_normalized = sitk.GetImageFromArray(corrected_np_normalized)
    corrected_image_normalized.CopyInformation(corrected_image_full_resolution)
    print("Z-score normalization applied successfully.")
except Exception as e:
    print(f"An error occurred during normalization: {e}")
    raise

# Step 7: Compare and Print Statistics
try:
    print("\nComparing intensity statistics between original and corrected images...")

    original_stats = sitk.StatisticsImageFilter()
    corrected_stats = sitk.StatisticsImageFilter()

    original_stats.Execute(image)
    corrected_stats.Execute(corrected_image_normalized)

    stats_comparison = {
        'Original': {
            'Mean': original_stats.GetMean(),
            'StdDev': original_stats.GetSigma(),
            'Min': original_stats.GetMinimum(),
            'Max': original_stats.GetMaximum()
        },
        'Corrected': {
            'Mean': corrected_stats.GetMean(),
            'StdDev': corrected_stats.GetSigma(),
            'Min': corrected_stats.GetMinimum(),
            'Max': corrected_stats.GetMaximum()
        }
    }

    for image_type, stats in stats_comparison.items():
        print(f"\n{image_type} Image Statistics:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.3f}")

    mean_change = ((stats_comparison['Corrected']['Mean'] - stats_comparison['Original']['Mean']) /
                   stats_comparison['Original']['Mean'] * 100)
    std_change = ((stats_comparison['Corrected']['StdDev'] - stats_comparison['Original']['StdDev']) /
                  stats_comparison['Original']['StdDev'] * 100)

    print(f"\nChanges after correction:")
    print(f"  Mean change: {mean_change:.1f}%")
    print(f"  StdDev change: {std_change:.1f}%")

except Exception as e:
    print(f"Error comparing intensity statistics: {e}")
    raise

# Step 8: Save the Corrected and Normalized Image
try:
    sitk.WriteImage(corrected_image_normalized, PATHS['output_image'])
    print(f"Corrected and normalized image saved at: {PATHS['output_image']}")
except Exception as e:
    print(f"Error saving corrected and normalized image: {e}")
    raise

# Print total execution time
end_time = time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
