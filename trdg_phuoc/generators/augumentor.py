from PIL import Image
import numpy as np

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.geometric.transforms import *
import random
import cv2
import random
import numpy as np
import cv2
import numpy as np
from PIL import Image, ImageOps
import random
# from augraphy import *



class Stretch:
	def __init__(self, p, rng=None):
		self.rng = np.random.default_rng() if rng is None else rng
		self.tps = cv2.createThinPlateSplineShapeTransformer()
		self.prob = p

	def __call__(self, img):
		if self.rng.uniform(0, 1) > self.prob:
			return img
		if isinstance(img, np.ndarray):
			img = Image.fromarray(img)
		w, h = img.size
		img = np.asarray(img)
		srcpt = []
		dstpt = []

		w_33 = 0.33 * w
		w_50 = 0.50 * w
		w_66 = 0.66 * w

		h_50 = 0.50 * h

		p = 0

		b = [.2, .3]
		frac = random.choice(b)  # Randomly select frac like Curve uses random.choice for rmin

		# left-most
		srcpt.append([p, p])
		srcpt.append([p, h - p])
		srcpt.append([p, h_50])
		x = self.rng.uniform(0, frac) * w_33
		dstpt.append([p + x, p])
		dstpt.append([p + x, h - p])
		dstpt.append([p + x, h_50])

		# 2nd left-most
		srcpt.append([p + w_33, p])
		srcpt.append([p + w_33, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		dstpt.append([p + w_33 + x, p])
		dstpt.append([p + w_33 + x, h - p])

		# 3rd left-most
		srcpt.append([p + w_66, p])
		srcpt.append([p + w_66, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		dstpt.append([p + w_66 + x, p])
		dstpt.append([p + w_66 + x, h - p])

		# right-most
		srcpt.append([w - p, p])
		srcpt.append([w - p, h - p])
		srcpt.append([w - p, h_50])
		x = self.rng.uniform(-frac, 0) * w_33
		dstpt.append([w - p + x, p])
		dstpt.append([w - p + x, h - p])
		dstpt.append([w - p + x, h_50])

		n = len(dstpt)
		matches = [cv2.DMatch(i, i, 0) for i in range(n)]
		dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
		src_shape = np.asarray(srcpt).reshape((-1, n, 2))
		self.tps.estimateTransformation(dst_shape, src_shape, matches)
		img = self.tps.warpImage(img, borderValue=(255, 255, 255))
		# img = Image.fromarray(img)

		return img


class Distort:
	def __init__(self, p, rng=None):
		self.rng = np.random.default_rng() if rng is None else rng
		self.tps = cv2.createThinPlateSplineShapeTransformer()
		self.prob = p

	def __call__(self, img):
		if self.rng.uniform(0, 1) > self.prob:
			return img
		if isinstance(img, np.ndarray):
			img = Image.fromarray(img)
		w, h = img.size
		img = np.asarray(img)
		srcpt = []
		dstpt = []

		w_33 = 0.33 * w
		w_50 = 0.50 * w
		w_66 = 0.66 * w

		h_50 = 0.50 * h

		p = 0

		b = [.2, .3]
		frac = random.choice(b)  # Randomly select frac like Curve uses random.choice for rmin

		# top pts
		srcpt.append([p, p])
		x = self.rng.uniform(0, frac) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([p + x, p + y])

		srcpt.append([p + w_33, p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([p + w_33 + x, p + y])

		srcpt.append([p + w_66, p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([p + w_66 + x, p + y])

		srcpt.append([w - p, p])
		x = self.rng.uniform(-frac, 0) * w_33
		y = self.rng.uniform(0, frac) * h_50
		dstpt.append([w - p + x, p + y])

		# bottom pts
		srcpt.append([p, h - p])
		x = self.rng.uniform(0, frac) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([p + x, h - p + y])

		srcpt.append([p + w_33, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([p + w_33 + x, h - p + y])

		srcpt.append([p + w_66, h - p])
		x = self.rng.uniform(-frac, frac) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([p + w_66 + x, h - p + y])

		srcpt.append([w - p, h - p])
		x = self.rng.uniform(-frac, 0) * w_33
		y = self.rng.uniform(-frac, 0) * h_50
		dstpt.append([w - p + x, h - p + y])

		n = len(dstpt)
		matches = [cv2.DMatch(i, i, 0) for i in range(n)]
		dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
		src_shape = np.asarray(srcpt).reshape((-1, n, 2))
		self.tps.estimateTransformation(dst_shape, src_shape, matches)
		img = self.tps.warpImage(img, borderValue=(255, 255, 255))
		# img = Image.fromarray(img)

		return img


class Curve:
	def __init__(self, p, square_side=224, rng=None):
		self.tps = cv2.createThinPlateSplineShapeTransformer()
		self.side = square_side
		self.rng = np.random.default_rng() if rng is None else rng
		self.prob = p

	def __call__(self, img):
		if self.rng.uniform(0, 1) > self.prob:
			return img
		if isinstance(img, np.ndarray):
			img = Image.fromarray(img)
		orig_w, orig_h = img.size

		if orig_h != self.side or orig_w != self.side:
			img = img.resize((self.side, self.side), Image.BICUBIC)

		isflip = self.rng.uniform(0, 1) > 0.5
		if isflip:
			img = ImageOps.flip(img)

		img = np.asarray(img)
		w = self.side
		h = self.side
		w_25 = 0.25 * w
		w_50 = 0.50 * w
		w_75 = 0.75 * w

		b = [1.1, .95, .8]
		rmin = random.choice(b)

		r = self.rng.uniform(rmin, rmin + .1) * h
		x1 = (r ** 2 - w_50 ** 2) ** 0.5
		h1 = r - x1

		t = self.rng.uniform(0.4, 0.5) * h

		w2 = w_50 * t / r
		hi = x1 * t / r
		h2 = h1 + hi

		sinb_2 = ((1 - x1 / r) / 2) ** 0.5
		cosb_2 = ((1 + x1 / r) / 2) ** 0.5
		w3 = w_50 - r * sinb_2
		h3 = r - r * cosb_2

		w4 = w_50 - (r - t) * sinb_2
		h4 = r - (r - t) * cosb_2

		w5 = 0.5 * w2
		h5 = h1 + 0.5 * hi
		h_50 = 0.50 * h

		srcpt = [(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h),
				 (0, h_50), (w, h_50)]
		dstpt = [(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4),
				 (w - w4, h4), (w5, h5), (w - w5, h5)]

		n = len(dstpt)
		matches = [cv2.DMatch(i, i, 0) for i in range(n)]
		dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
		src_shape = np.asarray(srcpt).reshape((-1, n, 2))
		self.tps.estimateTransformation(dst_shape, src_shape, matches)
		img = self.tps.warpImage(img, borderValue=(255, 255, 255))
		
		img = img.resize((orig_w, orig_h), Image.BICUBIC)
		return img



class WrapAug(ImageOnlyTransform):
	def __init__(self, p):
		super().__init__(p = 1)
		self.p_distort = p[0]
		self.p_strech = p[1]
		self.p_cur = p[2]
		self.augs = [ Curve(p=self.p_cur),Stretch(p=self.p_strech), Distort(p=self.p_distort)]
	
	def apply(self, img,**kwags):
		for aug in self.augs:
			img = aug(img)
		return img

class RandomDottedLine(ImageOnlyTransform):
	def __init__(self, num_lines=1, p=1):
		super(RandomDottedLine, self).__init__(p=p)
		self.num_lines = num_lines

	def apply(self, img, **params):
		self.img = img
		if random.random() < 0.5:
			self.random_setup_horizal()
			self.img = self.main_draw(self.img)
		if random.random() < 0.35:
			self.random_left_vertical()
			self.img = self.main_draw(self.img)
		if random.random() < 0.35:
			self.random_right_vertical()
			self.img = self.main_draw(self.img)
		return self.img

	def main_draw(self,img):
		
		if self.line_type != "solid":
			self._draw_dotted_line(
				img, (self.x1, self.y1), (self.x2, self.y2), self.color, self.thickness, self.line_type
			)
		else:
			cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), self.color, self.thickness)
		return img
	def random_generally(self):
		self.color = tuple(np.random.randint(20, 50, size=3).tolist())
		self.thickness = np.random.randint(5,8)
		self.line_type = random.choice(["solid"])
	def random_setup_horizal(self):
		h, w = self.img.shape[:2]
		max_height = int(0.6*h)

		max_left = int(0.15*w)
		max_right = int(0.85*w)
		
		self.x1, self.y1 = np.random.randint(0, max_left), np.random.randint(max_height, h)
		# self.x2, self.y2 = np.random.randint(max_right, w), np.random.randint(self.y1 - 0.05*h, max(h,self.y1 + 0.05,h))
		self.x2, self.y2 = np.random.randint(max_right, w), self.y1
		self.random_generally()
  
	def random_left_vertical(self):
	 
		h, w = self.img.shape[:2]
		max_width = int(0.1*w)
		max_top = int(0.2*h)
		max_bottom = int(0.8*h)
  
		self.x1, self.y1 = np.random.randint(0, max_width), np.random.randint(0,max_top)
		self.x2, self.y2 = self.x1, np.random.randint(max_bottom, h)
		self.random_generally()
  
	def random_right_vertical(self):
	 
		h, w = self.img.shape[:2]
		max_width = int(0.9*w)
		max_top = int(0.2*h)
		max_bottom = int(0.8*h)
  
		self.x1, self.y1 = np.random.randint(max_width, w), np.random.randint(0,max_top)
		self.x2, self.y2 = self.x1, np.random.randint(max_bottom, h)
		self.random_generally()
  
	def _draw_dotted_line(self, img, pt1, pt2, color, thickness, line_type):
		# Calculate the distance between the points
		dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
		# Number of segments
		num_segments = max(int(dist // 5), 1)
		# Generate points along the line
		x_points = np.linspace(pt1[0], pt2[0], num_segments)
		y_points = np.linspace(pt1[1], pt2[1], num_segments)
		# Draw segments
		for i in range(num_segments - 1):
			if line_type == "dotted" and i % 2 == 0:
				pt_start = (int(x_points[i]), int(y_points[i]))
				pt_end = (int(x_points[i]), int(y_points[i]))
				cv2.circle(img, pt_start, thickness, color, -1)
			elif line_type == "dashed" and i % 4 < 2:
				pt_start = (int(x_points[i]), int(y_points[i]))
				pt_end = (int(x_points[i + 1]), int(y_points[i + 1]))
				cv2.line(img, pt_start, pt_end, color, thickness)
		return img

	def get_transform_init_args_names(self):
		return ("num_lines",)
	
class ImgAugTransformV2:
	def __init__(self,mode = 'train'):
		# self.secound_aug  = LightingGradient(light_position=None,
		# 									  direction=None,
		# 									  max_brightness=255,
		# 									  min_brightness=0,
		# 									  mode="gaussian",
		# 									  )
		self.aug = A.Compose(
				[
		

			
				A.OneOf([
						A.AdditiveNoise(noise_type="gaussian",
								spatial_mode="per_pixel",
								noise_params={"mean_range": (0.0, 0.0), "std_range": (0.02, 0.02)} ,
								p =1),
						A.GridDropout(ratio = 0.2,unit_size_range  = [8,10], fill = random.randint(0,255),p = 1),
						A.PixelDropout(dropout_prob=0.2,drop_value=random.randint(0,255),p=1),
				
					], p=1),
	
				# A.RandomBrightnessContrast(brightness_limit = [-0.15,0.55],contrast_limit  =[ 0.3,0.4],p = 1),
				# A.RandomBrightnessContrast(brightness_limit = [-0.15,0.2],p = 1),
				A.ColorJitter(brightness  = 0,contrast=0,saturation  = 0.7,hue  = 0.7,p = 0.7),
				
				# A.RandomGamma(p= 1,gamma_limit=[50, 130]),
				A.OneOf([
			 			A.GaussianBlur(sigma = 1,blur_limit=17, p= 1 ),
						A.MotionBlur(blur_limit=[13, 17],p = 	1),
						A.Defocus(p = 1,radius=[11, 15])
				],p = 1),
					
				]
			)

		self.equalize = None
		assert mode in ['test', 'train'] , "mode must in test  or train"
		self.mode = mode
		
		
	def __call__(self, img):
		img = np.asarray(img,dtype=np.uint8)
  
		alpha = img[:,:,-1]
		img = img[:,:,:-1]
		
		transformed = self.aug(image = img)
		img = transformed["image"]
		
		rgba_array = np.dstack((np.array(img), alpha))  # Stack RGB with alpha
		img = Image.fromarray(rgba_array, mode='RGBA')
		return img