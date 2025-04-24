import cv2
import numpy as np

# Paths
image_path = "/work/21013187/phuoc/TextRecognitionDataGenerator2/outs/760.5_1.jpg"
txt_path =  image_path.replace(".jpg",'.txt')

# Read polygons from text file
polygons = []
with open(txt_path, 'r') as f:
    for line in f:
        # Assuming format: "(x1,y1) (x2,y2) (x3,y3) (x4,y4)" per line
        # Remove parentheses and split into coordinate pairs
        coords = line.strip().replace('(', '').replace(')', '').split()
        coords = [coord.replace(',','') for coord in coords]
        # Convert to list of tuples
        polygon = [
            (int(coords[0]), int(coords[1])),  # top-left
            (int(coords[2]), int(coords[3])),  # top-right
            (int(coords[4]), int(coords[5])),  # bottom-right
            (int(coords[6]), int(coords[7]))   # bottom-left
        ]
        polygons.append(polygon)

# Load the image
img = cv2.imread(image_path)

# Draw each polygon
for polygon in polygons:
    # Convert polygon points to numpy array for OpenCV
    pts = np.array(polygon, np.int32)
    pts = pts.reshape((-1, 1, 2))
    # Draw the polygon in red
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

# Save the result
cv2.imwrite("test.png", img)

