import os
from model import carotidSegmentation, analyze_plaque, highlight_and_analyze_plaque
# from analysis import analyze_carotid_plaque, visualize_results
import cv2
from stack import load_image_slices,generate_3d_models
# from mmm import plot_pred
from visual import main as visualizer

def main():
    model = carotidSegmentation()
    
    # image_loc = 'data/Common Carotid Artery Ultrasound Images/US images/202202071357530054VAS_slice_557.png'
    image_loc = 'data/DATASET1/frame-04.jpg'
    
    pred_img_path = model.plot_prediction(image_loc,image_name='extracted')
    # pred_img_path = model.plot_pred(image_loc)
    analyze_plaque(pred_img_path)
    highlight_and_analyze_plaque(pred_img_path)

# Use map_location to load the model on CPU
#model = torch.load("model.pth", map_location=torch.device('cpu'))

if __name__ == "__main__":
    main()