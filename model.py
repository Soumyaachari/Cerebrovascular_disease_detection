import numpy as np
import torch
import matplotlib.pyplot as plt
from unet import UNet
from utils import DiceLoss, load_config
from torchvision import transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from utils import DiceLoss
from datetime import datetime
from stack import save_slice_as_image

#self.net.load_state_dict(torch.load(f'models/{model}.pth', map_location=torch.device('cpu')))
class carotidSegmentation():
    """Class for loading and generating predictions via trained segmentation model"""
    def __init__(self, model='unet_1'):
        config_path = './config/'
        config_file = f'{model}.yaml'
        config = load_config(config_file, config_path)
        batch_size = 1
        
        self._image_transforms = torch.nn.Sequential(
            transforms.CenterCrop(512),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        
        self.net = UNet(
            in_channels=config['in_channels'], 
            n_classes=config['n_classes'], 
            depth=config['depth'], 
            batch_norm=config['batch_norm'], 
            padding=config['padding'], 
            up_mode=config['up_mode'],
        )
        self.net.load_state_dict(torch.load(f'models/{model}.pth',map_location=torch.device('cpu')))
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
    
    def plot_prediction_old(self, image_loc, labels=False, text=True, image_name=None):
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()
        background = image[0][2].detach().numpy()
        plt.imshow(background, cmap='Greys_r', alpha=1)
        plt.imshow(pred_out, 
                cmap='YlOrRd',
                alpha=pred_out*.5)

        # masked_pred_out = np.ma.masked_where(pred_out == 0, pred_out)
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(background, cmap='hot', interpolation='nearest', origin='upper')
        plt.title('Heatmap of Pixel Density')
        plt.colorbar(label='Pixel Intensity')
        plt.show()
        
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
    
    def plot_prediction(self, image_loc, labels=False, text=True, image_name=None):
        image = self.get_image(image_loc)
        preds = self.predict(image_loc)
        pred_out = preds[0][0].detach().numpy()  # Extract the prediction mask
        background = image[0][2].detach().numpy()  # Extract the background image

        # Define a threshold for "high" values in pred_out
        threshold = 0.2  # Adjust this threshold based on what you consider "masked"

        # Create a mask for areas where pred_out is above the threshold
        high_value_mask = pred_out >= threshold

        # Extract the corresponding area in the background where pred_out is high
        extracted_background = np.where(high_value_mask, background, np.nan)  # Use np.nan for unmasked areas

        # Plot the original background
        plt.figure(figsize=(10, 8))
        plt.imshow(background, cmap='Greys_r')
        plt.title('Original Image')
        plt.show()

        # save_slice_as_image(background,'generated',image_name)


        # Plot the extracted area
        plt.figure(figsize=(10, 8))
        plt.imshow(extracted_background, cmap='Greys_r')
        plt.title('Carotid Artery')
        plt.colorbar(label='Pixel Intensity')
        plt.show()

        # save_slice_as_image(extracted_background,'generated',image_name)

        # Plot the heatmap
        # plt.figure(figsize=(10, 8))
        # plt.imshow(extracted_background, cmap='hot', interpolation='nearest', origin='upper')
        # plt.title('Heatmap of Pixel Density')
        # plt.colorbar(label='Pixel Intensity')
        # plt.show()

        return extracted_background  # Return the extracted area if you want to use it further
    
def analyze_plaque(extracted_background):
    # Remove NaN values from the array for analysis
    valid_pixels = extracted_background[~np.isnan(extracted_background)]

    # Calculate basic statistics
    mean_intensity = np.mean(valid_pixels)
    median_intensity = np.median(valid_pixels)
    std_dev_intensity = np.std(valid_pixels)

    # Print the analysis
    print("\n\n\nPixel Intensity Analysis of Carotid Artery Region:")
    print(f"Mean Intensity: {mean_intensity:.2f}")
    print(f"Median Intensity: {median_intensity:.2f}")
    print(f"Standard Deviation: {std_dev_intensity:.2f}")

    # Plot histogram for pixel intensity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(valid_pixels, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Pixel Intensity Distribution in Carotid Artery Region")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Plaque detection based on intensity
    threshold = mean_intensity + 1.5 * std_dev_intensity  # Define a threshold for high intensity
    plaque_pixels = valid_pixels[valid_pixels > threshold]

    # Calculate plaque thickness as the ratio of high-intensity pixels to total pixels
    plaque_thickness_ratio = len(plaque_pixels) / len(valid_pixels)
    estimated_plaque_thickness = plaque_thickness_ratio * 100  # Convert to percentage

    # Print plaque analysis
    print("\n\n\nPlaque Analysis:")
    print(f"Plaque Detection Intensity Threshold: {threshold:.2f}")
    print(f"Estimated Plaque Thickness (Percentage of High-Intensity Pixels): {estimated_plaque_thickness:.2f}%")

    # Return results for further use if needed
    return {
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "std_dev_intensity": std_dev_intensity,
        "plaque_thickness_ratio": plaque_thickness_ratio,
        "threshold": threshold,
    }

def highlight_and_analyze_plaque(extracted_background):
    # Remove NaN values and define threshold for plaque detection
    valid_pixels = extracted_background[~np.isnan(extracted_background)]
    mean_intensity = np.mean(valid_pixels)
    std_dev_intensity = np.std(valid_pixels)
    threshold = mean_intensity + 0.001 * std_dev_intensity  # Threshold for plaque

    threshold = 30
    # Create a mask for the plaque regions (where intensity is above threshold)
    plaque_mask = (extracted_background >= threshold)
    
    # Calculate the area of the artery and plaque
    total_artery_area = np.sum(~np.isnan(extracted_background))  # Total artery area in pixels
    plaque_area = np.sum(plaque_mask)  # Plaque area in pixels

    # Convert areas to percentage
    plaque_percentage = (plaque_area / total_artery_area) * 100

    # Print area analysis
    print("\n\n\nArea Analysis:")
    print(f"Total Artery Area (in pixels): {total_artery_area}")
    print(f"Plaque Area (in pixels): {plaque_area}")
    print(f"Plaque Percentage of Artery: {plaque_percentage:.2f}%")

    # Visualize the plaque regions on the extracted background
    plt.figure(figsize=(10, 8))
    plt.imshow(extracted_background, cmap='Greys_r')
    plt.imshow(np.where(plaque_mask, extracted_background, np.nan), cmap='Blues', alpha=0.6)
    plt.title("Carotid Artery with Plaque Regions Highlighted")
    plt.colorbar(label="Pixel Intensity")
    # plt.savefig('./success2', bbox_inches='tight')
    plt.show()

    # save_slice_as_image(plaque_area,'generated',image_name)




    # Return the analysis results if needed
    return {
        "total_artery_area": total_artery_area,
        "plaque_area": plaque_area,
        "plaque_percentage": plaque_percentage,
        "threshold": threshold,
    }




if __name__ == "__main__":
    image_loc = 'data/Common Carotid Artery Ultrasound Images/US images/202201121748100022VAS_slice_2319.png'
    seg_model = carotidSegmentation()
    test_out = seg_model.predict(image_loc)
    print(test_out.shape)
    seg_model.plot_pred(image_loc, labels=True)