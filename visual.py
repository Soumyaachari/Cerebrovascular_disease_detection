# import os
# import cv2
# import numpy as np
# import pyvista as pv

# # Set the folder path containing 2D images
# folder_path = r"C:\Users\vikas\OneDrive\Desktop\3dmodel\blendedimage"  # Replace with your folder path

# # Get the list of image files and sort them to maintain correct slice order
# image_files = sorted(os.listdir(folder_path))

# # Initialize a list to store resized image slices
# image_slices = []

# # Load the first image to determine target dimensions
# first_image_path = os.path.join(folder_path, image_files[0])
# first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
# target_size = first_image.shape  # Use the shape of the first image as the target

# # Process each image file and resize to the target dimensions
# for filename in image_files:
#     slice_path = os.path.join(folder_path, filename)
#     slice_image = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

#     # Resize the slice to the target size if necessary
#     if slice_image.shape != target_size:
#         slice_image = cv2.resize(slice_image, (target_size[1], target_size[0]))  # Resize to match the target

#     image_slices.append(slice_image)

# # Stack all slices into a 3D volume
# volume = np.stack(image_slices, axis=-1)

# # Print the shape of the volume to confirm it's 3D
# print("Volume shape:", volume.shape)

# # Create a pyvista grid by wrapping the numpy array
# grid = pv.wrap(volume)

# # Plot the 3D volume using pyvista
# plotter = pv.Plotter()
# opacity = [0, 0, 0.1, 0.3, 0.6, 0.9, 1]  # Customize opacity levels for different intensity values

# plotter.add_volume(grid, cmap="gray", opacity=opacity, shade=True)
# plotter.show()
import os
import cv2
import numpy as np
import pyvista as pv

def main():

    # Set the folder path containing 2D images
    folder_path = f"data/Common Carotid Artery Ultrasound Images/Expert mask images"  # Replace with your folder path

    # Get the list of image files and sort them to maintain correct slice order
    image_files = sorted(os.listdir(folder_path))

    # Initialize a list to store resized image slices
    image_slices = []

    # Load the first image to determine target dimensions
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    target_size = first_image.shape  # Use the shape of the first image as the target

    # Process each image file and resize to the target dimensions
    for filename in image_files[:15]:
        slice_path = os.path.join(folder_path, filename)
        slice_image = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

        # Resize the slice to the target size if necessary
        if slice_image.shape != target_size:
            slice_image = cv2.resize(slice_image, (target_size[1], target_size[0]))  # Resize to match the target

        image_slices.append(slice_image)

    # Stack all slices into a 3D volume
    volume = np.stack(image_slices, axis=-1)

    # Convert the 3D volume array to a PyVista object and apply spacing
    grid = pv.wrap(volume)

    # Scale the Z-axis to simulate 1mm spacing between each slice
    # Adjust the scaling factors as necessary to achieve the correct spacing
    grid.scale([2, 2, 2])  # Set scaling for X, Y, and Z (1 unit for Z represents 1mm)

    # Plot the 3D volume using PyVista
    plotter = pv.Plotter()
    opacity = [0, 0, 0.1, 0.3, 0.6, 0.9, 1]  # Customize opacity levels for different intensity values

    plotter.add_volume(grid, cmap="gray", opacity=opacity, shade=True)
    plotter.show()

if __name__ == '__main__':
    main()