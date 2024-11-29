import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, filters
from datetime import datetime

class CarotidAnalyzer:
    """Analyzes carotid artery boundaries and plaque from segmentation predictions"""
    
    def __init__(self):
        # Standard measurements (in pixels, adjustable based on image resolution)
        self.normal_imt = 8  # Normal intima-media thickness
        self.normal_lumen = 200  # Normal lumen area
        
        # Plaque detection thresholds
        self.plaque_thickness_threshold = 12  # Threshold for plaque thickness
        self.intensity_threshold = 0.7  # Threshold for high-intensity regions
        
    def detect_boundaries_and_plaque(self, pred_mask, original_image):
        """Detects boundaries and identifies plaque regions"""
        # Convert prediction to binary mask
        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # Find main boundaries
        contours = measure.find_contours(binary_mask, 0.5)
        contours.sort(key=lambda x: len(x), reverse=True)
        
        outer_boundary = contours[0] if contours else None
        
        # Find inner boundary
        eroded = morphology.erosion(binary_mask, morphology.disk(5))
        inner_contours = measure.find_contours(eroded, 0.5)
        inner_contours.sort(key=lambda x: len(x), reverse=True)
        inner_boundary = inner_contours[0] if inner_contours else None
        
        # Detect plaque regions
        plaque_regions = self._detect_plaque(original_image, binary_mask, outer_boundary, inner_boundary)
        
        return outer_boundary, inner_boundary, plaque_regions
    
    def _detect_plaque(self, original_image, mask, outer_boundary, inner_boundary):
        """Detect plaque regions based on wall thickness and intensity"""
        if outer_boundary is None or inner_boundary is None:
            return None
            
        # Create empty plaque mask
        plaque_mask = np.zeros_like(mask, dtype=np.float32)
        
        # Calculate distance map from inner to outer boundary
        distance_map = ndimage.distance_transform_edt(mask)
        
        # Identify regions with increased thickness
        thick_regions = distance_map > self.plaque_thickness_threshold
        
        # Identify high-intensity regions in the original image
        normalized_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))
        high_intensity = normalized_image > self.intensity_threshold
        
        # Combine thickness and intensity criteria
        plaque_regions = thick_regions & high_intensity & mask.astype(bool)
        
        return plaque_regions
    
    def calculate_metrics(self, outer_boundary, inner_boundary, plaque_regions):
        """Calculate metrics including plaque measurements"""
        if outer_boundary is None or inner_boundary is None:
            return None
        
        # Basic measurements
        outer_area = self._polygon_area(outer_boundary)
        inner_area = self._polygon_area(inner_boundary)
        wall_area = outer_area - inner_area
        
        # Plaque measurements
        plaque_area = np.sum(plaque_regions) if plaque_regions is not None else 0
        plaque_burden = (plaque_area / wall_area * 100) if wall_area > 0 else 0
        
        # Wall thickness measurements
        max_thickness, avg_thickness = self._calculate_wall_thickness(outer_boundary, inner_boundary)
        
        # Calculate stenosis
        stenosis = (1 - inner_area / self.normal_lumen) * 100 if self.normal_lumen > 0 else 0
        
        return {
            'wall_area': wall_area,
            'lumen_area': inner_area,
            'plaque_area': plaque_area,
            'plaque_burden': plaque_burden,
            'max_thickness': max_thickness,
            'avg_thickness': avg_thickness,
            'stenosis_percentage': stenosis
        }
    
    def assess_risk(self, metrics):
        """Enhanced risk assessment including plaque characteristics"""
        if metrics is None:
            return "Unable to assess risk - boundaries not detected properly"
        
        risk_factors = []
        risk_score = 0
        
        # Assess plaque burden
        if metrics['plaque_burden'] > 40:
            risk_factors.append(f"Severe plaque burden: {metrics['plaque_burden']:.1f}%")
            risk_score += 3
        elif metrics['plaque_burden'] > 20:
            risk_factors.append(f"Moderate plaque burden: {metrics['plaque_burden']:.1f}%")
            risk_score += 2
        
        # Assess wall thickness
        if metrics['max_thickness'] > self.normal_imt * 2:
            risk_factors.append(f"Severe wall thickening: {metrics['max_thickness']:.1f} pixels")
            risk_score += 3
        elif metrics['max_thickness'] > self.normal_imt * 1.5:
            risk_factors.append(f"Moderate wall thickening: {metrics['max_thickness']:.1f} pixels")
            risk_score += 2
        
        # Assess stenosis
        if metrics['stenosis_percentage'] > 70:
            risk_factors.append(f"Severe stenosis: {metrics['stenosis_percentage']:.1f}%")
            risk_score += 3
        elif metrics['stenosis_percentage'] > 50:
            risk_factors.append(f"Moderate stenosis: {metrics['stenosis_percentage']:.1f}%")
            risk_score += 2
        
        # Determine overall risk level
        if risk_score >= 6:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
            
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors
        }
    
    def plot_analysis(self, image, pred_mask, save_path=None):
        """Enhanced plotting with plaque visualization"""
        outer_boundary, inner_boundary, plaque_regions = self.detect_boundaries_and_plaque(pred_mask, image)
        metrics = self.calculate_metrics(outer_boundary, inner_boundary, plaque_regions)
        risk_assessment = self.assess_risk(metrics)
        
        plt.figure(figsize=(12, 8))
        
        # Plot original image
        plt.imshow(image, cmap='gray')
        
        # Plot boundaries
        if outer_boundary is not None:
            plt.plot(outer_boundary[:, 1], outer_boundary[:, 0], 'r-', label='Outer Wall', linewidth=2)
        if inner_boundary is not None:
            plt.plot(inner_boundary[:, 1], inner_boundary[:, 0], 'b-', label='Inner Wall', linewidth=2)
        
        # Highlight plaque regions
        if plaque_regions is not None:
            plt.imshow(plaque_regions, cmap='OrRd', alpha=0.3, label='Plaque')
        
        plt.title('Carotid Artery Analysis with Plaque Detection')
        plt.legend()
        
        # Add detailed metrics and risk assessment
        if metrics:
            info_text = (
                f"MEASUREMENTS:\n"
                f"Wall Area: {metrics['wall_area']:.1f} px²\n"
                f"Plaque Area: {metrics['plaque_area']:.1f} px²\n"
                f"Plaque Burden: {metrics['plaque_burden']:.1f}%\n"
                f"Max Wall Thickness: {metrics['max_thickness']:.1f} px\n"
                f"Stenosis: {metrics['stenosis_percentage']:.1f}%\n\n"
                f"RISK ASSESSMENT:\n"
                f"Risk Level: {risk_assessment['risk_level']}\n"
                f"Risk Score: {risk_assessment['risk_score']}\n"
                f"Risk Factors:\n" + '\n'.join(f"- {factor}" for factor in risk_assessment['risk_factors'])
            )
            plt.figtext(0.02, 0.02, info_text, bbox=dict(facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
    
    def _calculate_wall_thickness(self, outer_boundary, inner_boundary):
        """Calculate maximum and average wall thickness"""
        distances = []
        for point in outer_boundary:
            dist = np.min(np.sqrt(np.sum((inner_boundary - point) ** 2, axis=1)))
            distances.append(dist)
        return np.max(distances), np.mean(distances)
    
    def _polygon_area(self, boundary):
        """Calculate area of a polygon using shoelace formula"""
        x = boundary[:, 1]
        y = boundary[:, 0]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def analyze_carotid_image(seg_model, image_loc):
    """Complete analysis pipeline"""
    # Get image and prediction
    image = seg_model.get_image(image_loc)
    pred = seg_model.predict(image_loc)
    
    # Convert to numpy arrays
    image_np = image[0][2].detach().numpy()
    pred_np = pred[0][0].detach().numpy()
    
    # Run analysis
    analyzer = CarotidAnalyzer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'carotid_analysis_{timestamp}.png'
    
    analysis_path = analyzer.plot_analysis(image_np, pred_np, save_path)
    return analysis_path