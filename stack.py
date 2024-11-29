import numpy as np
import os
from skimage.io import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import matplotlib.pyplot as plt

def save_slice_as_image(slice_2d, output_folder, slice_index : str):
    """
    Save a 2D slice as an image file.
    
    Parameters:
    - slice_2d: 2D numpy array, the slice image to be saved.
    - output_folder: str, the folder path to save the image.
    - slice_index: int, index of the slice (used for naming).
    """
    os.makedirs(output_folder, exist_ok=True)
    plt.imshow(slice_2d, cmap='gray')
    plt.axis('off')  # Turn off axis
    slice_filename = os.path.join(output_folder, f"slice_{slice_index}.png")
    plt.savefig(slice_filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def load_image_slices(folder_path):
    """
    Load a series of 2D image slices from a folder and stack them into a 3D numpy array.
    
    Parameters:
    - folder_path: str, path to the folder containing the 2D image files.
    
    Returns:
    - volume_data: numpy.ndarray, a 3D array created by stacking the 2D image slices.
    """
    # List image files in the folder and sort to maintain order
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')])

    # Initialize a list to hold 2D arrays
    slices = []
    
    # Load each image and convert it to a 2D numpy array
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        slice_2d = imread(image_path, as_gray=True)  # Load as grayscale
        slices.append(slice_2d)

    # Stack all 2D slices into a single 3D numpy array (depth, height, width)
    volume_data = np.stack(slices, axis=0)

    return volume_data


def generate_3d_models(image_slices, threshold):
    # Convert image slices to numpy arrays if not already in that format
    volume_data = np.array(image_slices)
    
    # Create a mask for plaque regions
    plaque_mask = volume_data >= threshold

    # Original volume: original image slices
    original_volume = np.copy(volume_data)

    # Extracted volume without plaque
    plaque_free_volume = np.where(plaque_mask, np.nan, volume_data)

    # Extracted volume with plaque regions highlighted
    plaque_highlighted_volume = np.copy(volume_data)
    plaque_highlighted_volume[plaque_mask] = np.nan  # For masking other areas, only plaque remains
    
    # Plotting
    fig = plt.figure(figsize=(18, 6))

    # Original model
    ax1 = fig.add_subplot(131, projection='3d')
    z, y, x = np.nonzero(~np.isnan(original_volume))
    ax1.scatter(x, y, z, c=original_volume[~np.isnan(original_volume)], cmap='Blues', marker='o', alpha=0.5)
    ax1.set_title("Original Artery Model")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Plaque-free model
    ax2 = fig.add_subplot(132, projection='3d')
    z_free, y_free, x_free = np.nonzero(~np.isnan(plaque_free_volume))
    ax2.scatter(x_free, y_free, z_free, c=plaque_free_volume[~np.isnan(plaque_free_volume)], cmap='Greys', marker='o', alpha=0.5)
    ax2.set_title("Plaque-Free Artery Model")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Plaque-highlighted model
    ax3 = fig.add_subplot(133, projection='3d')
    # Show the full artery in a light color
    ax3.scatter(x, y, z, c=original_volume[~np.isnan(original_volume)], cmap='Blues', marker='o', alpha=0.1)
    # Overlay plaque regions in red
    z_plaque, y_plaque, x_plaque = np.nonzero(plaque_mask)
    ax3.scatter(x_plaque, y_plaque, z_plaque, c=volume_data[plaque_mask], cmap='Reds', marker='o', alpha=0.6)
    ax3.set_title("Artery Model with Plaque Highlighted")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    # plt.tight_layout()
    # plt.show()
    plt.savefig('./success')

