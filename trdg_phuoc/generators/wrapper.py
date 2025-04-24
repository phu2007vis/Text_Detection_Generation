import cv2
import numpy as np
import os
import json
from PIL import Image

def puttext(image,text = "ga"):
	font = cv2.FONT_HERSHEY_SIMPLEX
	text_position = (10, 50)  # 10 pixels from left, 50 pixels from top
	font_scale = 1.0
	color = (0, 255, 0)  # Green text in BGR
	thickness = 2

	image = cv2.putText(image, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)
	return image

class WrapImage:
	def __init__(self,image_dir,label_file):
		self.image_dir = image_dir
		self.image_names =[]
		self.map_point = {}
	 
		with open(label_file,'r',encoding = 'utf8') as f:
			data = f.readlines()
			for line in data:
				line = line.strip().split('\t')
				img_name = line[0]
				line =  json.loads(line[1])[0]['points']
				self.map_point[img_name]  = line
				self.image_names.append(img_name)
		self._setup_wh_map()
		self._aug_map_point = {}
				
	def __len__(self):
		return len(self.map_point.keys())
	def __getitem__(self,idx):
			img_name = self.image_names[idx]
			img_path = os.path.join(self.image_dir,img_name)
			original_img = cv2.imread(img_path)
		
			original_img = cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
			self.fetch_data(original_img,self.map_point[img_name])
			self.crop()
			self._aug_map_point[img_name]= self.augment_points(self.map_point[img_name])
			return self.warped
	def put_image_back(self, warped_img, idx, warped_mask=None):
		# import pdb;pdb.set_trace()
		# cv2.imwrite("wrap_img.jpg",np.asarray(warped_img, dtype=np.uint8))
		# exit()
		# Convert PIL Images to numpy arrays if necessary, preserving original data
		warped_img_array = np.copy(np.asarray(warped_img, dtype=np.uint8)) if isinstance(warped_img, Image.Image) else np.copy(warped_img)
		warped_mask_array = np.copy(np.asarray(warped_mask, dtype=np.uint8)) if warped_mask is not None and isinstance(warped_mask, Image.Image) else warped_mask

		# Load original image
		img_name = self.image_names[idx]
		img_path = os.path.join(self.image_dir, img_name)
		original_img = cv2.imread(img_path)
		original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
		
		# Process image
		self.fetch_data(original_img, self._aug_map_point[img_name], warped_img_array)
		self.put()
		result_image = Image.fromarray(np.copy(self.image))  # Create a copy to avoid modifying original
		
		if warped_mask_array is not None:
			# Process mask separately with a fresh copy of original shape
			original_mask = np.zeros_like(original_img)
			self.fetch_data(original_mask, self._aug_map_point[img_name], warped_mask_array)
			self.put()
			result_mask = Image.fromarray(np.copy(self.image))  # Create a copy to avoid modifying original
			return result_image, result_mask
		
		return result_image, None
	def augment_points(self, points, max_noise=20, max_angle=15, scale_range=(0.8, 1.2)):
		"""
		Augment the source points with random noise, rotation, and variable scaling.

		:param points: List or array of 4 points (e.g., [[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
		:param max_noise: Maximum pixel offset for random noise (default: 10)
		:param max_angle: Maximum rotation angle in degrees (default: 15)
		:param scale_range: Tuple (min_scale, max_scale) for random scaling (default: (0.9, 1.1))
		:return: Augmented and clipped points as a NumPy array with shape (4, 2)
		"""
		# Validate input
		points = np.array(points, dtype=np.float32)
		if points.shape != (4, 2):
			raise ValueError(f"Expected 4 points with 2D coordinates, got shape {points.shape}")

		# Get image dimensions
		if self.image is None:
			raise ValueError("Image not set; cannot determine boundaries for clipping")
		h, w = self.image.shape[:2]

		# Step 1: Add random noise to each point
		noise = np.random.uniform(-max_noise, max_noise, points.shape)
		points = points + noise

		# Step 2: Rotate points around their centroid by a random angle
		angle = np.random.uniform(-max_angle, max_angle)
		centroid = np.mean(points, axis=0)
		theta = np.radians(angle)
		cos_theta, sin_theta = np.cos(theta), np.sin(theta)
		points_shifted = points - centroid
		rotated_x = points_shifted[:, 0] * cos_theta - points_shifted[:, 1] * sin_theta
		rotated_y = points_shifted[:, 0] * sin_theta + points_shifted[:, 1] * cos_theta
		points = np.column_stack((rotated_x, rotated_y)) + centroid

		# Step 3: Apply random scaling (anisotropic)
		scale_x = np.random.uniform(scale_range[0], scale_range[1])
		scale_y = np.random.uniform(scale_range[0], scale_range[1])
		points = centroid + np.array([scale_x, scale_y]) * (points - centroid)

		# Step 4: Clip points to stay within image boundaries
		points[:, 0] = np.clip(points[:, 0], 0, w - 1)
		points[:, 1] = np.clip(points[:, 1], 0, h - 1)

		return points.astype(np.float32)
	def put(self):
		# Ensure we have the required data
		if self.warped_img is None or self.image is None or self.points is None:
			raise ValueError("Warped image, original image, or points not set. Call fetch_data first.")

		# Get dimensions
		h, w = self.image.shape[:2]
		
		# Source points (original points in the image)
		src_pts = np.array(self.points, np.float32)
		# Destination points (rectangle corners of the warped image)
		warped_h, warped_w = self.warped_img.shape[:2]
		dst_pts = np.array([
			[0, 0],              # top-left
			[0, warped_h - 1],   # bottom-left
			[warped_w - 1, warped_h - 1],  # bottom-right
			[warped_w - 1, 0]    # top-right
		], dtype=np.float32)

		# Compute inverse perspective transform
		M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

		# Warp the modified image back to original perspective
		warped_back = cv2.warpPerspective(self.warped_img, M_inv, (w, h))

		# Create mask for the region
		mask = np.zeros((h, w), dtype=np.uint8)
		cv2.fillConvexPoly(mask, src_pts.astype(np.int32), 255)

		# Invert mask
		mask_inv = cv2.bitwise_not(mask)

		# Combine images using masks
		img_bg = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
		img_fg = cv2.bitwise_and(warped_back, warped_back, mask=mask)
		self.image = cv2.add(img_bg, img_fg)

	def fetch_data(self, image, points, warped_img=None):
		self.image = np.copy(image)  # Ensure we work with a copy
		self.points = points
		self.warped_img = np.copy(warped_img) if warped_img is not None else None
	def crop(self):
		points = self.points
		image = self.image
		# Convert points to numpy array
		src_pts = np.array(points, np.float32)
		
		# Calculate the width and height of the destination rectangle
		# Distance between points to determine the rectangle dimensions
		width1 = np.sqrt(((src_pts[0][0] - src_pts[1][0]) ** 2) + ((src_pts[0][1] - src_pts[1][1]) ** 2))
		width2 = np.sqrt(((src_pts[2][0] - src_pts[3][0]) ** 2) + ((src_pts[2][1] - src_pts[3][1]) ** 2))
		height1 = np.sqrt(((src_pts[0][0] - src_pts[3][0]) ** 2) + ((src_pts[0][1] - src_pts[3][1]) ** 2))
		height2 = np.sqrt(((src_pts[1][0] - src_pts[2][0]) ** 2) + ((src_pts[1][1] - src_pts[2][1]) ** 2))
		
		# Use the maximum width and height to ensure the polygon fits
		width = max(int(width1), int(width2))
		height = max(int(height1), int(height2))
		
		# Define the destination points (a perfect rectangle)
		dst_pts = np.array([
			[0, 0],           # top-left
			[0, height - 1],  # bottom-left
			[width - 1, height - 1],  # bottom-right
			[width - 1, 0]    # top-right
		], dtype=np.float32)
		
		# Compute the perspective transform matrix
		M = cv2.getPerspectiveTransform(src_pts, dst_pts)
		
		# Apply the perspective transform to the image
		self.warped = cv2.warpPerspective(image, M, (width, height))
	def _setup_wh_map(self):
		self.wh_mapper = {}
		for img_name in self.image_names:
			img_path = os.path.join(self.image_dir, img_name)
			original_img = cv2.imread(img_path)
			h,w,_ = original_img.shape
			self.wh_mapper[img_name] = (h,w)
	def convert_point(self, points, idx, from_warped_to_original=True):
		
		"""
		Convert one or more points between warped and original image coordinates
		Args:
			points: Tuple (x, y) or List[Tuple(x, y)] - single point or list of points to convert
			idx: int - index of the image in image_names
			from_warped_to_original: bool - True to convert from warped to original,
										False to convert from original to warped
		Returns:
			List[Tuple(int, int)] - list of converted point coordinates
		"""
		img_name = self.image_names[idx]
		h, w = self.wh_mapper[img_name]
		
		
		src_pts = np.array(self._aug_map_point[img_name], np.float32)
		
		
		if hasattr(self, 'warped') and self.warped is not None:
			warped_h, warped_w = self.warped.shape[:2]
		else:
			warped_h, warped_w = h, w
		
		dst_pts = np.array([
			[0, 0],
			[0, warped_h - 1],
			[warped_w - 1, warped_h - 1],
			[warped_w - 1, 0]
		], dtype=np.float32)

		M = cv2.getPerspectiveTransform(dst_pts, src_pts) if from_warped_to_original else cv2.getPerspectiveTransform(src_pts, dst_pts)
		
		# Handle both single point and list of points
		if isinstance(points, tuple) and len(points) == 2:  # Single point
			point_array = np.array([[points]], dtype=np.float32)
		else:  # List of points
			point_array = np.array(points, dtype=np.float32).reshape(-1, 2)
			point_array = point_array[:, np.newaxis, :]  # Reshape to (n, 1, 2)
		
		converted = cv2.perspectiveTransform(point_array, M)
		
		# Convert to list of integer tuples
		return [(int(x), int(y)) for x, y in converted.reshape(-1, 2)]
# Example usage:
if __name__ == "__main__":
	# Load image
	img_foler = r"C:\Users\9999\Downloads\phuoc_test"
	txt_path = r"C:\Users\9999\Downloads\phuoc_test\Label.txt"
	wraper = WrapImage(img_foler,txt_path)
	img = wraper[0]
	cv2.imwrite("ga.jpg",img)
	img  = puttext(img,'ga')
	new_img = wraper.put_image_back(img,0)
	cv2.imwrite("ga2.jpg",new_img)
	
