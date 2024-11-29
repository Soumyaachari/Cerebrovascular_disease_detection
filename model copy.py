import os
import json
import numpy as np
import pandas as pd
import argparse
import torch
from skimage.measure import label  # Ensure you import 'label' from skimage.measure
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage import img_as_ubyte

import nibabel as nib
import matplotlib.pyplot as plt
from dataset import CarotidDataset
from skimage.measure import regionprops

from unet import UNet
from utils import DiceLoss, load_config
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from utils import DiceLoss
from datetime import datetime
from skimage import filters, morphology, measure
from scipy.spatial import distance
from PIL import Image


class carotidSegmentation():
    """Class for loading and generating predictions via trained segmentation model"""
    def __init__(self, model='unet_1'):
        config_path='./config/'
        config_file=f'{model}.yaml'
        config = load_config(config_file, config_path)
        batch_size = 1
        
        self._image_transforms = torch.nn.Sequential(
            transforms.CenterCrop(512),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )
        
        self.net = UNet(
            in_channels=config['in_channels'], 
            n_classes=config['n_classes'], 
            depth=config['depth'], 
            batch_norm=config['batch_norm'], 
            padding=config['padding'], 
            up_mode=config['up_mode'])
        self.net.load_state_dict(torch.load(f'models/{model}.pth'))
        self.net.eval()
    
    def get_image(self,image_loc):
        image = read_image(image_loc)
        image = image.float()
        image = image.unsqueeze(0)
        image = self._image_transforms(image)
        return image

    def get_label(self, image_loc):
        mask_loc = image_loc.replace("US images", "Expert mask images")
        mask = read_image(mask_loc)
        mask = mask.float()
        mask = mask.unsqueeze(0)
        mask = self._image_transforms(mask)
        mask = mask > 0
        mask = mask.type(torch.int8)
        
        return mask

    def predict(self,image_loc):
        image = self.get_image(image_loc)
        preds = self.net(image)
        return preds

    def eval(self, image_loc):
        pred = self.predict(image_loc)
        label = self.get_label(image_loc)
        loss = DiceLoss()
        loss_out = loss(pred, label)
        return loss_out.item()
    
    def plot_pred(self, image_loc, labels=False, text=True, image_name=None):
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()
        plt.imshow(background, cmap='Greys_r', alpha=1)
        plt.imshow(pred_out, 
                cmap='YlOrRd',
                alpha=pred_out*.5)
        if labels != False:
            label_out = self.get_label(image_loc)
            label_out = label_out[0][0]#.numpy()
            plt.imshow(label_out, 
            cmap='RdYlBu', 
            alpha=label_out*.5)
            if text == True:
                dice_loss = round(self.eval(image_loc), 4)
                plt.xlabel(f'Prediction = Red, True Label = Blue \n Dice Loss: {dice_loss}')
        else:
            if text == True:
                plt.xlabel('Prediction = Red',)
        if text == True:
            plt.title('Carotid Artery Segmentation')
        plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
        
        # plt.show()

        if not image_name:
            dt = datetime.now().strftime(f"%d-%m-%Y")
            save_path = f'{dt}.png'
        else:
            save_path = f'{image_name}.png'
        plt.savefig(save_path, bbox_inches='tight')
        
        return f'{save_path}.png'
    

    def save_shaded_area(self, image_loc, save_path):
        # Get the image and prediction
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        
        # Extract the predicted mask (shaded portion)
        pred_out = preds[0][0].detach().numpy()
        
        # Create a new figure to plot only the shaded portion
        plt.figure(figsize=(8, 8))
        
        # Display only the shaded portion (predicted mask)
        plt.imshow(pred_out, cmap='YlOrRd', alpha=pred_out * 0.5)
        
        # Add a title or any additional text you want on the plot
        plt.title('Shaded Portion of the Segmentation')
        
        # Remove axis ticks and labels for a clean image
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        
        # Save the plot as an image file
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        
        # Optionally close the plot to free memory
        plt.close()

    def save_shaded_region_only(self, image_loc, save_path):
    # Get the original image and predicted mask
        image = self.get_image(image_loc)  # Original image
        preds = self.predict(image_loc)    # Prediction (segmentation mask)
        
        # Extract the predicted mask (shaded portion)
        pred_out = preds[0][0].detach().numpy()  # Prediction mask

        # Convert the image into a NumPy array (if it's a tensor or a different format)
        image_array = image[0][2].detach().numpy()  # Original image

        # Create a mask where the predicted area is 1 and the rest is 0
        mask = pred_out > 0.5  # Assuming the mask is a binary mask (adjust threshold as needed)
        
        # Apply the mask to the original image, setting non-masked regions to black (or another color)
        extracted_image = np.zeros_like(image_array)  # Create an empty (black) image

        # Use the mask to extract the corresponding parts of the original image
        extracted_image[mask] = image_array[mask]
        
        # Save the extracted region as an image
        extracted_image_pil = Image.fromarray(np.uint8(extracted_image))  # Convert to PIL Image for saving
        extracted_image_pil.save(save_path)
        
        # Optionally, display the extracted region
        plt.imshow(extracted_image)
        plt.title('Extracted Region')
        plt.axis('off')
        plt.show()


    def shade_area_by_density(self, image_loc, save_path):
        # Get the original image and predicted mask (density information)
        image = self.get_image(image_loc)  # Original image
        preds = self.predict(image_loc)    # Prediction (segmentation mask)
        
        # Extract the predicted mask (density information)
        pred_out = preds[0][0].detach().numpy()  # Prediction mask (density)
        
        # Convert the original image into a NumPy array (if it's a tensor or a different format)
        image_array = image[0][2].detach().numpy()  # Original image (assumed grayscale, shape: (512, 512))

        # Normalize the predicted mask (to ensure it ranges from 0 to 1)
        pred_out_normalized = (pred_out - np.min(pred_out)) / (np.max(pred_out) - np.min(pred_out))
        
        # Expand the original image to have 3 channels (RGB)
        image_array_expanded = np.stack([image_array] * 3, axis=-1)  # Shape becomes (512, 512, 3)

        # Create a shaded version of the image by blending the original image with a color based on pixel density
        # We use the mask density to blend the original image with a shade (using a color map, e.g., YlOrRd)
        shaded_image = image_array_expanded * (1 - pred_out_normalized[..., np.newaxis]) + \
                    (pred_out_normalized[..., np.newaxis] * 255)  # Blend using density as the weight
        
        # Ensure the shaded image is in the range [0, 255]
        shaded_image = np.clip(shaded_image, 0, 255)

        # Save the shaded image
        shaded_image_pil = Image.fromarray(np.uint8(shaded_image))  # Convert to PIL Image for saving
        shaded_image_pil.save(save_path)
        
        # Optionally, display the shaded image
        plt.imshow(shaded_image.astype(np.uint8))
        plt.title('Shaded by Pixel Density')
        plt.axis('off')
        plt.show()

    def plot_pred3(self, image_loc, labels=False, text=True, image_name=None):
        # Get the original image and the predictions
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        
        # Prediction mask (shaded region)
        pred_out = preds[0][0].detach().numpy()
        
        # Background image (the original image)
        background = image[0][2].detach().numpy()

        # Define a threshold to detect non-zero (shaded) areas in the prediction
        threshold = 0.1  # Adjust this based on your requirements (small density regions)
        mask = pred_out > threshold  # Binary mask of the shaded areas (non-zero regions)

        # Find bounding box of the non-zero areas (shaded region)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        row_start, row_end = np.where(rows)[0][[0, -1]]
        col_start, col_end = np.where(cols)[0][[0, -1]]
        
        # Crop the image and prediction to the bounding box
        cropped_background = background[row_start:row_end + 1, col_start:col_end + 1]
        cropped_pred_out = pred_out[row_start:row_end + 1, col_start:col_end + 1]
        
        # Plot the cropped shaded region
        plt.imshow(cropped_background, cmap='Greys_r', alpha=1)
        plt.imshow(cropped_pred_out, cmap='YlOrRd', alpha=cropped_pred_out * 0.5)

        if labels:
            label_out = self.get_label(image_loc)
            label_out = label_out[0][0].detach().numpy()  # Assuming the labels are 2D
            cropped_label_out = label_out[row_start:row_end + 1, col_start:col_end + 1]
            plt.imshow(cropped_label_out, cmap='RdYlBu', alpha=cropped_label_out * 0.5)
            
            if text:
                dice_loss = round(self.eval(image_loc), 4)
                plt.xlabel(f'Prediction = Red, True Label = Blue \n Dice Loss: {dice_loss}')
        else:
            if text:
                plt.xlabel('Prediction = Red')

        if text:
            plt.title('Carotid Artery Segmentation')

        # Remove ticks and labels
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        
        # Generate the save path
        if not image_name:
            dt = datetime.now().strftime("%d-%m-%Y")
            save_path = f'{dt}_shaded.png'
        else:
            save_path = f'{image_name}_shaded.png'
        
        # Save the cropped image
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the plot to avoid displaying

        return f'{save_path}'

    def plot_shaded_region_last(self, image_loc, labels=False, text=True, image_name=None):
        # Load image and predictions
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()
        
        # Threshold the predicted output to identify shaded regions (e.g., regions with high values)
        threshold = np.mean(pred_out) + np.std(pred_out)  # Threshold to define shaded regions
        shaded_region = pred_out > threshold  # Binary mask for shaded regions
        
        # Label the connected regions
        labeled_regions = label(shaded_region)  # Use 'label' correctly here
        
        # Find the largest region
        regions = regionprops(labeled_regions)  # Now using the correct 'regionprops' function
        largest_region = max(regions, key=lambda r: r.area)  # Largest shaded region by area
        
        # Get the bounding box of the largest region
        minr, minc, maxr, maxc = largest_region.bbox
        cropped_image = background[minr:maxr, minc:maxc]
        
        # Create a variation map based on pixel values in the shaded region
        variation_map = np.abs(pred_out - background)  # Pixel variation (absolute difference)
        variation_map = variation_map[minr:maxr, minc:maxc]  # Crop the variation map to the region
        
        # Normalize the variation map to [0, 1] if needed
        variation_map_min = np.min(variation_map)
        variation_map_max = np.max(variation_map)
        
        # Normalize to the range [0, 1]
        if variation_map_max != variation_map_min:
            variation_map = (variation_map - variation_map_min) / (variation_map_max - variation_map_min)
        
        # Convert the variation map to uint8 (0-255 range)
        variation_map_image = img_as_ubyte(variation_map)  # Now it will be in [0, 255] range
        
        # Plot the cropped image and variation map
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the cropped region
        axes[0].imshow(cropped_image, cmap='gray')
        axes[0].set_title('Cropped Shaded Region')
        axes[0].axis('off')
        
        # Plot the variation map
        axes[1].imshow(variation_map_image, cmap='coolwarm')
        axes[1].set_title('Pixel Variation in Region')
        axes[1].axis('off')
        
        # Add labels if required
        if labels:
            label_out = self.get_label(image_loc)
            label_out = label_out[0][0].detach().numpy()
            axes[0].imshow(label_out[minr:maxr, minc:maxc], cmap='RdYlBu', alpha=0.5)
            if text:
                dice_loss = round(self.eval(image_loc), 4)
                axes[0].set_xlabel(f'Prediction = Red, True Label = Blue \n Dice Loss: {dice_loss}')
        
        # Add title if required
        if text:
            plt.suptitle('Shaded Region and Pixel Variation')
        
        # Save the images
        if not image_name:
            dt = datetime.now().strftime(f"%d-%m-%Y")
            save_path = f'{dt}_shaded_region.png'
        else:
            save_path = f'{image_name}_shaded_region.png'
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        
        # Save the variation map image
        variation_map_save_path = save_path.replace('_shaded_region', '_variation_map')
        plt.imsave(variation_map_save_path, variation_map_image, cmap='coolwarm')
        
        return save_path, variation_map_save_path
    

    def predict_plaque_thickness(self, image_loc, text=True, image_name=None):
        # Load image and predictions
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()

        # Define a threshold to detect high-density regions (potential plaque)
        threshold = np.mean(pred_out) + np.std(pred_out) * 2  # Adjust multiplier based on expected plaque intensity
        plaque_region = pred_out > threshold  # Binary mask for potential plaque

        # Label connected regions to identify plaque areas
        labeled_regions = label(plaque_region)
        regions = regionprops(labeled_regions)

        plaque_thickness = []
        pixel_density = []
        risk_level = []

        for region in regions:
            # Calculate the plaque's area and bounding box
            area = region.area
            minr, minc, maxr, maxc = region.bbox

            # Calculate the pixel density of the region (mean intensity)
            plaque_pixels = pred_out[minr:maxr, minc:maxc]
            density = np.mean(plaque_pixels)  # Average pixel intensity in the plaque region

            # Estimate plaque thickness based on the bounding box width (difference in column indices)
            thickness = maxc - minc  # Horizontal width of the plaque region

            # Assign a risk factor based on density and thickness
            if density > 0.8 and thickness > 15:  # High density and large thickness
                risk = 'High Risk'
            elif density > 0.6 and thickness > 10:  # Medium density and moderate thickness
                risk = 'Medium Risk'
            else:  # Low density or small thickness
                risk = 'Low Risk'

            plaque_thickness.append(thickness)
            pixel_density.append(density)
            risk_level.append(risk)

            # Plotting the plaque region on the image
            plt.figure(figsize=(8, 6))
            plt.imshow(background, cmap='gray')
            plt.title(f'Plaque Risk: {risk} (Density: {density:.2f}, Thickness: {thickness}px)')
            plt.axis('off')

            # Optionally display the bounding box around the plaque region
            plt.gca().add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                                            linewidth=2, edgecolor='red', facecolor='none'))

            # Save the image with the plaque highlighted
            if not image_name:
                dt = datetime.now().strftime(f"%d-%m-%Y")
                save_path = f'{dt}_plaque_risk.png'
            else:
                save_path = f'{image_name}_plaque_risk.png'

            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        # Return a summary of the risk levels, thickness, and pixel density
        plaque_summary = {
            "Plaque Thickness": plaque_thickness,
            "Pixel Density": pixel_density,
            "Risk Levels": risk_level
        }

        return plaque_summary, save_path
    
    def plot_with_contours(self, image_loc, contour_levels=10, image_name=None):
        """
        This method generates contour lines based on pixel intensity and overlays them on the image.
        
        Parameters:
        - image_loc: The location of the image file.
        - contour_levels: Number of contour levels (lines) to be generated.
        - image_name: Optional name for the saved output image.
        
        Returns:
        - save_path: Path to the saved image with contour lines.
        """
        # Load image and predictions
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()

        # Plot the background image
        plt.figure(figsize=(8, 8))
        plt.imshow(background, cmap='gray', alpha=1)
        
        # Add contour lines based on the intensity of pred_out
        contours = plt.contour(pred_out, levels=contour_levels, colors='red', linewidths=0.5)
        plt.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')  # Add contour labels if desired

        # Set title and disable axis ticks
        plt.title('Carotid Artery Segmentation with Contour Lines')
        plt.axis('off')

        # Save the image with contours
        if not image_name:
            dt = datetime.now().strftime("%d-%m-%Y")
            save_path = f'{dt}_contours.png'
        else:
            save_path = f'{image_name}_contours.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_with_dual_contours(self, image_loc, low_intensity_level=0.2, high_intensity_level=0.8, image_name=None):
        """
        This method adds dual contour lines based on pixel intensity, one for low intensity and one for high intensity.
        
        Parameters:
        - image_loc: The location of the image file.
        - low_intensity_level: Relative threshold for low-intensity contour lines (0 to 1 scale).
        - high_intensity_level: Relative threshold for high-intensity contour lines (0 to 1 scale).
        - image_name: Optional name for the saved output image.
        
        Returns:
        - save_path: Path to the saved image with dual contour lines.
        """
        # Load image and predictions
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()

        # Normalize pred_out to [0, 1] range
        pred_out_normalized = (pred_out - pred_out.min()) / (pred_out.max() - pred_out.min())

        # Define actual intensity levels based on the normalized pixel intensity
        low_threshold = np.percentile(pred_out_normalized, low_intensity_level * 100)
        high_threshold = np.percentile(pred_out_normalized, high_intensity_level * 100)

        # Plot the background image
        plt.figure(figsize=(8, 8))
        plt.imshow(background, cmap='gray', alpha=1)

        # Add contour lines for low intensity regions
        low_contours = plt.contour(pred_out_normalized, levels=[low_threshold], colors='blue', linewidths=0.5)
        plt.clabel(low_contours, inline=True, fontsize=8, fmt='%1.2f')  # Label contours if desired

        # Add contour lines for high intensity regions
        high_contours = plt.contour(pred_out_normalized, levels=[high_threshold], colors='red', linewidths=1)
        plt.clabel(high_contours, inline=True, fontsize=8, fmt='%1.2f')  # Label contours if desired

        # Set title and remove axis ticks
        plt.title('Carotid Artery Segmentation with Dual Contour Lines')
        plt.axis('off')

        # Save the image with dual contours
        if not image_name:
            dt = datetime.now().strftime("%d-%m-%Y")
            save_path = f'{dt}_dual_contours.png'
        else:
            save_path = f'{image_name}_dual_contours.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        return save_path
    

    def plot_contours_within_shaded(self, image_loc, low_intensity_level=0.2, high_intensity_level=0.8, image_name=None):
        """
        This method adds contour lines only within the shaded (high-density) region based on pixel intensity.
        
        Parameters:
        - image_loc: The location of the image file.
        - low_intensity_level: Relative threshold for low-intensity contour lines (0 to 1 scale).
        - high_intensity_level: Relative threshold for high-intensity contour lines (0 to 1 scale).
        - image_name: Optional name for the saved output image.
        
        Returns:
        - save_path: Path to the saved image with contour lines within shaded region.
        """
        # Load image and predictions
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()

        # Normalize pred_out to [0, 1] range
        pred_out_normalized = (pred_out - pred_out.min()) / (pred_out.max() - pred_out.min())

        # Define an initial shaded region mask based on a threshold
        initial_threshold = np.percentile(pred_out_normalized, 75)  # Modify percentile as necessary
        shaded_region_mask = pred_out_normalized > initial_threshold

        # Apply mask to only focus contours inside the shaded region
        masked_pred_out = np.where(shaded_region_mask, pred_out_normalized, 0)

        # Define low and high thresholds based on the masked area
        low_threshold = np.percentile(masked_pred_out[shaded_region_mask], low_intensity_level * 100)
        high_threshold = np.percentile(masked_pred_out[shaded_region_mask], high_intensity_level * 100)

        # Plot the background image
        plt.figure(figsize=(8, 8))
        plt.imshow(background, cmap='gray', alpha=1)

        # Add contour lines for low intensity within the shaded region
        low_contours = plt.contour(masked_pred_out, levels=[low_threshold], colors='blue', linewidths=0.5)
        plt.clabel(low_contours, inline=True, fontsize=8, fmt='%1.2f')  # Label contours if desired

        # Add contour lines for high intensity within the shaded region
        high_contours = plt.contour(masked_pred_out, levels=[high_threshold], colors='red', linewidths=1)
        plt.clabel(high_contours, inline=True, fontsize=8, fmt='%1.2f')  # Label contours if desired

        # Set title and remove axis ticks
        plt.title('Contour Lines Within Shaded Region')
        plt.axis('off')

        # Save the image with contours inside the shaded region
        if not image_name:
            dt = datetime.now().strftime("%d-%m-%Y")
            save_path = f'{dt}_contours_within_shaded.png'
        else:
            save_path = f'{image_name}_contours_within_shaded.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_cropped_contours_within_shaded(self, image_loc, low_intensity_level=0.2, high_intensity_level=0.8, image_name=None):
        """
        This method adds contour lines based on pixel intensity only within the initial shaded region and crops the rest.
        
        Parameters:
        - image_loc: The location of the image file.
        - low_intensity_level: Relative threshold for low-intensity contour lines (0 to 1 scale).
        - high_intensity_level: Relative threshold for high-intensity contour lines (0 to 1 scale).
        - image_name: Optional name for the saved output image.
        
        Returns:
        - save_path: Path to the saved image with cropped contour lines within shaded region.
        """
        # Load image and predictions
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()

        # Normalize pred_out to [0, 1] range
        pred_out_normalized = (pred_out - pred_out.min()) / (pred_out.max() - pred_out.min())

        # Define an initial shaded region mask based on a threshold
        initial_threshold = np.percentile(pred_out_normalized, 75)  # Modify percentile as necessary
        shaded_region_mask = pred_out_normalized > initial_threshold

        # Generate bounding box coordinates for the shaded region
        labeled_shaded_region = label(shaded_region_mask)
        region = max(regionprops(labeled_shaded_region), key=lambda r: r.area)  # Select the largest region
        min_row, min_col, max_row, max_col = region.bbox

        # Crop the background and pred_out to the bounding box of the shaded region
        cropped_background = background[min_row:max_row, min_col:max_col]
        cropped_pred_out = pred_out_normalized[min_row:max_row, min_col:max_col]
        cropped_mask = shaded_region_mask[min_row:max_row, min_col:max_col]

        # Apply mask to cropped prediction for contouring only within the shaded area
        masked_pred_out = np.where(cropped_mask, cropped_pred_out, 0)

        # Define low and high thresholds based on the cropped and masked area
        low_threshold = np.percentile(masked_pred_out[cropped_mask], low_intensity_level * 100)
        high_threshold = np.percentile(masked_pred_out[cropped_mask], high_intensity_level * 100)

        # Plot the cropped background image
        plt.figure(figsize=(8, 8))
        plt.imshow(cropped_background, cmap='gray', alpha=1)

        # Add contour lines for low intensity within the cropped shaded region
        low_contours = plt.contour(masked_pred_out, levels=[low_threshold], colors='blue', linewidths=0.5)
        plt.clabel(low_contours, inline=True, fontsize=8, fmt='%1.2f')  # Label contours if desired

        # Add contour lines for high intensity within the cropped shaded region
        high_contours = plt.contour(masked_pred_out, levels=[high_threshold], colors='red', linewidths=1)
        plt.clabel(high_contours, inline=True, fontsize=8, fmt='%1.2f')  # Label contours if desired

        # Set title and remove axis ticks
        plt.title('Contour Lines Within Cropped Shaded Region')
        plt.axis('off')

        # Save the cropped image with contours
        if not image_name:
            dt = datetime.now().strftime("%d-%m-%Y")
            save_path = f'{dt}_cropped_contours_within_shaded.png'
        else:
            save_path = f'{image_name}_cropped_contours_within_shaded.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        return save_path




if __name__ == "__main__":
    image_loc = 'data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_2319.png'
    seg_model = carotidSegmentation()
    test_out = seg_model.predict(image_loc)
    print(test_out.shape)
    seg_model.plot_pred(image_loc, labels=True)