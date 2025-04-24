import cv2
import numpy as np

# Function to load image from path and apply adaptive thresholding
def process_image_from_path(input_path, output_path):
    try:
        # Load the image from file path in grayscale
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Failed to load image from path")

        # Apply adaptive thresholding
        # Parameters: (image, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        thresh = cv2.adaptiveThreshold(
            img,                    # Input image
            255,                    # Max pixel value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method
            cv2.THRESH_BINARY,      # Threshold type
            11,                     # Block size (must be odd)
            4                      # Constant subtracted from mean
        )

        # Save the thresholded image
        cv2.imwrite(output_path, thresh)
        print(f"Image processed and saved as {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    input_file = "/work/21013187/phuoc/Image_Captionning_Transformer/data2/testset_processced/0418/images/000_002484.jpg"  # e.g., "C:/images/input.jpg"
    output_file = "thresholded_image.jpg"
    
    process_image_from_path(input_file, output_file)