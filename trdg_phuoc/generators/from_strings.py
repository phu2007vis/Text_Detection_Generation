import os
from typing import List, Tuple
import random
import argparse	
import os
import sys
import numpy as np

folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(folder_path)

from trdg_phuoc.generators.wrapper import WrapImage
from trdg_phuoc.generators.augumentor import ImgAugTransformV2
from trdg_phuoc.utils import mask_to_polygons
from PIL import Image
from trdg_phuoc import computer_text_generator
from multiprocessing import Pool
from augumentor import ImgAugTransformV2
img_aug = ImgAugTransformV2()
def _generate_text_box(args: Tuple[int, int, int, str, str, str, int,float,int]) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
	"""Helper function to generate a single text box in a separate process."""
	i, max_text_height, bg_width, text, font, text_color, size ,random_angle,image_height= args
	
	# Generate text image
	image, mask = computer_text_generator.generate(
		text=text,
		font=font,
		text_color=text_color,
		font_size=size,
		mask_value=i + 1,
	)
	
	
	# Calculate optimal size
	MARGIN = 5
	MARGIN_Y = random.randint(int(image_height*0.2),int(image_height*0.35))
	# MARGIN_Y = 0
	aspect_ratio = image.size[0] / image.size[1]
	new_height = min(max_text_height, image.size[1])
	new_width = min(int(new_height * aspect_ratio), bg_width - 2 * MARGIN)
	
	if new_width <= 0 or new_height <= 0:
		return None, None, (None,None)
	
	
	
	# Random initial position (to be refined later)
	pos_x = random.randint(MARGIN, bg_width - new_width - MARGIN)
	pos_y = random.randint(MARGIN+MARGIN_Y, image_height - new_height - MARGIN)
	
	rotated_img = image.rotate(random_angle, expand=1)
	rotated_mask = mask.rotate(random_angle, expand=1)
 
	# rotated_img = img_aug(rotated_img)	
	
	return rotated_img, rotated_mask,(pos_x,pos_y)
class GeneratorFromStrings:
	"""Generator that uses a given list of strings to create text images."""

	def __init__(
		self,
		
		fonts: List[str],
		count,
		text_color: str = None,
		image_dir: str = os.path.join(
			os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "images"
		),
		background_dir: str = "",
		color_path: str = "",
		background_polygons: str = None,
	):
		self.count = count
		self.count = -1  # Unused but kept for potential future use
		self.fonts = fonts
		self.text_color = text_color
		self.image_dir = image_dir
		self.background_dir = background_dir
		self.background_polygons = background_polygons

		# Initialize character counts for random strings
		# self.num_chars = []
		# for i in range(1, 7):
		# 	if i in (1, 2):
		# 		continue
		# 	self.num_chars.extend([i] * self.number_each)

		# Load font colors
		with open(color_path, "r") as f:
			self.all_font_colors = [x.strip() for x in f.readlines()]

		self.generated_count = 0
		self.setup_wrapper()

	def _random_string(self, num_char = int) -> str:
		"""Generate a random string with the given number of characters."""
		digits = "0123456789"
		
		if num_char == 1:
			return random.choice(digits)

		# Decide if the string should contain a decimal/comma separator
		contain_separator = True if num_char < 6 else False
		max_before = num_char - 2 if contain_separator else num_char
		before_count = max_before
		after_count = num_char - before_count - 1 if contain_separator else 0

		before_digits = "".join(random.choice(digits) for _ in range(before_count))
		if random.random() < 1:
			before_digits = '-'+before_digits
			print(before_digits)
		after_digits = "".join(random.choice(digits) for _ in range(after_count))

		if contain_separator:
			separator = random.choice([".", ","])
			return f"{before_digits}{separator}{after_digits}"
	
		return before_digits
	
	def _random_string2(self) -> str:
		"""Generate a random string with the given number of characters."""
		digits = "0123456789"
		
		num_char = random.randint(3,6)
		contain_separator  = True
		max_before = num_char - 2 if contain_separator else num_char
		before_count = max_before
		after_count = num_char - before_count - 1 if contain_separator else 0

		before_digits = "".join(random.choice(digits) for _ in range(before_count))
		after_digits = "".join(random.choice(digits) for _ in range(after_count))
		if random.random() < 0.2:
			before_digits = '-'+before_digits
		if contain_separator:
			separator = random.choice([".", ","])
			return f"{before_digits}{separator}{after_digits}"
		return before_digits

	def __iter__(self):
		return self

	def __next__(self):
		return self.next()

	def _setup_random(self):
		"""Set up random parameters for the next generation."""
		self.size = random.randint(50, 160)
		self.font = random.choice(self.fonts)
		self.text = self._random_string2()
		self.text_color = random.choice(self.all_font_colors)
		self.random_angle = random.choice([0,90])+random.uniform(-4,4)

	def next(self):
		"""Generate the next image and label."""
		if self.generated_count > count:
			raise StopIteration

		self.generated_count += 1

		return (
			self.generate(),self.text
		)

	def setup_wrapper(self):
		"""Initialize the image wrapper."""
		self.wrapper = WrapImage(self.background_dir, self.background_polygons)
	def generate(self) -> Tuple[Image.Image, Image.Image, List]:
		"""Generate an image with text overlaid on a background with optimized placement in parallel."""
		# Configuration
		MAX_BOXES = random.randint(10,14)
		MARGIN = 5
		MAX_ATTEMPTS = 10  # Maximum placement attempts per box
		
		# Initialize background
		from time import time
		t = time()
		wrap_index = random.randint(0, len(self.wrapper) - 1)
		background_img = Image.fromarray(self.wrapper[wrap_index]).convert("RGBA")
		bg_width, bg_height = background_img.size
		background_mask = Image.new("RGB", (bg_width, bg_height), (0, 0, 0))
		# print(f"Background init time {time()-t}")
		occupied_areas = []
		all_polygons = []
		available_height = bg_height - 2 * MARGIN
		

		# Prepare arguments for parallel processing
		args = []
		for i in range(MAX_BOXES):
			self._setup_random()
			max_text_height = min(self.size, available_height // MAX_BOXES)
			args.append([1, max_text_height, bg_width, self.text, self.font, self.text_color, self.size,self.random_angle,available_height])

			
		t = time()
		# Parallelize text box generation
		with Pool() as pool:
			results = pool.map(_generate_text_box, args)
		# print(f"Genereate box time: {time()-t}")
		t1 = time()
		# Process results and place text boxes
		for resized_img, resized_mask,(pos_x, pos_y) in results:
			if resized_img is None or resized_mask is None :
				continue

			new_width, new_height = resized_img.size
			
			# Smart placement with limited attempts
			placed = False
			attempts = 0
			best_pos = None
			min_overlap = float('inf')
			t2 = time()
			while attempts < MAX_ATTEMPTS and not placed:
				current_box = (pos_x, pos_y, pos_x + new_width, pos_y + new_height)
				overlaps = sum(self._boxes_overlap_area(current_box, occupied) 
							  for occupied in occupied_areas)
				
				if overlaps == 0:  # Perfect placement
					placed = True
				elif overlaps < min_overlap:  # Better than previous attempts
					min_overlap = overlaps
					best_pos = (pos_x, pos_y)
				
				if not placed:
					pos_x = random.randint(MARGIN, bg_width - new_width - MARGIN)
					pos_y = random.randint(MARGIN, bg_height - new_height - MARGIN)
				attempts += 1
			t3 = time()
			# print(f"Attempting time {t3-t2}")
			# Use best position if no perfect spot found
			if not placed and best_pos:
				pos_x, pos_y = best_pos
				current_box = (pos_x, pos_y, pos_x + new_width, pos_y + new_height)
			elif not placed:
				continue
			t4 = time()
			# Apply the text and update tracking
			background_img.paste(resized_img, (pos_x, pos_y), resized_img)
			background_mask.paste(resized_mask, (pos_x, pos_y))
			occupied_areas.append(current_box)
			t5 = time()
			# Convert polygons to original coordinates
			polygons = mask_to_polygons(background_mask)
			# print(f"Time conver poligons: {time()-t5}")
			converted_polygons = [self.wrapper.convert_point(polygon, wrap_index, from_warped_to_original=True) for polygon in polygons]
		
			all_polygons.extend(converted_polygons)
			background_mask = Image.new("RGB", (bg_width, bg_height), (0, 0, 0))
			# print(f"Time past image: {t5-t4}")
			
		
		# print(f"Put image time {time()-t1}")
		# Finalize output
		final_image = background_img.convert("RGB")
		
		final_image,final_mask = self.wrapper.put_image_back(final_image, wrap_index)
		
		return final_image, all_polygons

	def _boxes_overlap_area(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> int:
		"""Calculate overlapping area between two boxes."""
		x1, y1, x2, y2 = box1
		x3, y3, x4, y4 = box2
		
		if x1 >= x4 or x3 >= x2 or y1 >= y4 or y3 >= y2:
			return 0
			
		x_overlap = min(x2, x4) - max(x1, x3)
		y_overlap = min(y2, y4) - max(y1, y3)
		return x_overlap * y_overlap


def parse_arguments():
	"""
	Parse the command line arguments of the program.
	"""

	parser = argparse.ArgumentParser(
		description="Generate synthetic text data for text recognition."
	)
	parser.add_argument(
		"--output_dir", type=str, nargs="?", help="The output directory", default="out/"
	)
	parser.add_argument(
		"-i",
		"--aug",
		default=False,
		 action="store_true",
	)
	
	parser.add_argument(
		"-l",
		"--number_each",
		type=int,
		nargs="?",
	)
	parser.add_argument(
		"-c",
		"--count",
		type=int,
		nargs="?",
		help="The number of images to be created.",
		default= -1
	)
	parser.add_argument(
		"-rs",
		"--random_sequences",
		action="store_true",
		help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
		default=False,
	)
	parser.add_argument(
		"-let",
		"--include_letters",
		action="store_true",
		help="Define if random sequences should contain letters. Only works with -rs",
		default=False,
	)
	parser.add_argument(
		"-num",
		"--include_numbers",
		action="store_true",
		help="Define if random sequences should contain numbers. Only works with -rs",
		default=False,
	)
	parser.add_argument(
		"-sym",
		"--include_symbols",
		action="store_true",
		help="Define if random sequences should contain symbols. Only works with -rs",
		default=False,
	)
	parser.add_argument(
		"-w",
		"--length",
		type=int,
		nargs="?",
		help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
		default=1,
	)
	parser.add_argument(
		"-r",
		"--random",
		action="store_true",
		help="Define if the produced string will have variable word count (with --length being the maximum)",
		default=False,
	)
	parser.add_argument(
		"-f",
		"--format",
		type=int,
		nargs="?",
		help="Define the height of the produced images if horizontal, else the width",
		default=32,
	)
	parser.add_argument(
		"-t",
		"--thread_count",
		type=int,
		nargs="?",
		help="Define the number of thread to use for image generation",
		default=1,
	)
	parser.add_argument(
		"-e",
		"--extension",
		type=str,
		nargs="?",
		help="Define the extension to save the image with",
		default="jpg",
	)
	parser.add_argument(
		"-k",
		"--skew_angle",
		type=int,
		nargs="?",
		help="Define skewing angle of the generated text. In positive degrees",
		default=0,
	)
	parser.add_argument(
		"-rk",
		"--random_skew",
		action="store_true",
		help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
		default=False,
	)
	parser.add_argument(
		"-wk",
		"--use_wikipedia",
		action="store_true",
		help="Use Wikipedia as the source text for the generation, using this parameter ignores -r, -n, -s",
		default=False,
	)
	parser.add_argument(
		"-bl",
		"--blur",
		type=int,
		nargs="?",
		help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
		default=0,
	)
	parser.add_argument(
		"-rbl",
		"--random_blur",
		action="store_true",
		help="When set, the blur radius will be randomized between 0 and -bl.",
		default=False,
	)
	parser.add_argument(
		"-b",
		"--background",
		type=int,
		nargs="?",
		help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image",
		default=3,
	)
	parser.add_argument(
	   
		"--background_dir",
		type=str,
		default='',
	   
		),
	parser.add_argument(
	   
		"--color_path",
		type=str,
		default='',
	   
		)
	parser.add_argument(
		"-hw",
		"--handwritten",
		action="store_true",
		help='Define if the data will be "handwritten" by an RNN',
	)
	parser.add_argument(
		"-na",
		"--name_format",
		type=int,
		help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
		default=0,
	)
	parser.add_argument(
		"-om",
		"--output_mask",
		type=int,
		help="Define if the generator will return masks for the text",
		default=0,
	)
	parser.add_argument(
		"-obb",
		"--output_bboxes",
		type=int,
		help="Define if the generator will return bounding boxes for the text, 1: Bounding box file, 2: Tesseract format",
		default=0,
	)
	parser.add_argument(
		"-d",
		"--distorsion",
		type=int,
		nargs="?",
		help="Define a distortion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
		default=0,
	)
	parser.add_argument(
		"-do",
		"--distorsion_orientation",
		type=int,
		nargs="?",
		help="Define the distortion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
		default=0,
	)
	parser.add_argument(
		"-wd",
		"--width",
		type=int,
		nargs="?",
		help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
		default=-1,
	)
	parser.add_argument(
		"-al",
		"--alignment",
		type=int,
		nargs="?",
		help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
		default=1,
	)
	parser.add_argument(
		"-or",
		"--orientation",
		type=int,
		nargs="?",
		help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
		default=0,
	)
	parser.add_argument(
		"-tc",
		"--text_color",
		type=str,
		nargs="?",
		help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
		default="#282828",
	)
	parser.add_argument(
		"-sw",
		"--space_width",
		type=float,
		nargs="?",
		help="Define the width of the spaces between words. 2.0 means twice the normal space width",
		default=1.0,
	)
	parser.add_argument(
		"-cs",
		"--character_spacing",
		type=int,
		nargs="?",
		help="Define the width of the spaces between characters. 2 means two pixels",
		default=0,
	)
   
	parser.add_argument(
		"-fi",
		"--fit",
		action="store_true",
		help="Apply a tight crop around the rendered text",
		default=False,
	)
	parser.add_argument(
		"-fd",
		"--font_dir",
		type=str,
		nargs="?",
		help="Define a font directory to be used",
		default= '/work/21013187/phuoc/TextRecognitionDataGenerator2/phuoc_fonts/'
	)
	parser.add_argument(
		"-id",
		"--image_dir",
		type=str,
		nargs="?",
		help="Define an image directory to use when background is set to image",
		default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images"),
	)
	parser.add_argument(
		"-ca",
		"--case",
		type=str,
		nargs="?",
		help="Generate upper or lowercase only. arguments: upper or lower. Example: --case upper",
	)
	parser.add_argument(
		"-dt", "--dict", type=str, nargs="?", help="Define the dictionary to be used"
	)
	parser.add_argument(
		"-ws",
		"--word_split",
		action="store_true",
		help="Split on words instead of on characters (preserves ligatures, no character spacing)",
		default=False,
	)
	parser.add_argument(
		
		"--background_polygons",
		type=str,
		nargs="?",
		help="Define the width of the strokes",
		default='/work/21013187/phuoc/TextRecognitionDataGenerator2/trdg_phuoc/Label.txt',
	)

	parser.add_argument('--remove_exsist', action='store_true')
	return parser.parse_args()
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Tuple, List
import json
import os
from pathlib import Path

if __name__ == "__main__":
	from tqdm import tqdm
	args = parse_arguments()
	output_dir = args.output_dir
	distorsion_type = args.distorsion
	background_dir = args.background_dir
	color_path = args.color_path
	background_polygons = args.background_polygons
	# print(f"Type of distortion: {distorsion_type}")
	count = args.count
	# Create the directory if it does not exist.
	import shutil
	if os.path.exists(output_dir) and args.remove_exsist:
		print("Removing existing output f{output_dir}")
		shutil.rmtree(output_dir)
	os.makedirs(output_dir, exist_ok=True)
 
	import glob
	
	fonts = glob.glob(f'{args.font_dir}*.ttf')
 
	generator = GeneratorFromStrings(
		fonts = fonts,
		count = count,
		background_dir = background_dir,
		color_path = color_path,
		background_polygons = background_polygons
	)
	
	
	augumentor = ImgAugTransformV2()


	Path(output_dir).mkdir(parents=True, exist_ok=True)
	pbar = tqdm(range(count),total=count)
	# Pre-open label file in append mode with buffer
	output_file = os.path.join(output_dir, "labels.txt")
	with open(output_file, "a", encoding="utf-8", buffering=8192) as f:
		# Process generator items
		for i, ((img, polygons), lbl) in enumerate(generator):
			pbar.update(1)
			if img is None:
				continue
				
			# Prepare image name and save
			lbl = lbl.replace(",", ".")
			image_name = f"{lbl}_{i}.jpg"
			image_path = os.path.join(output_dir, image_name)
			
			# Uncomment if augmentation is needed
			# if augumentor:
			#     img = augumentor(img)
			
			img.save(image_path)
			
			# Optimize polygon processing
			# Convert to set of tuples directly (faster than dict.fromkeys)
			unique_coords = list({tuple(map(tuple, poly)) for poly in polygons})
			
			# Build label dictionary efficiently
			all_label_dict = [
				{
					"transcription": "hehe",
					"points": [list(point) for point in group],  # Convert tuples to lists
					"difficult": False
				}
				for group in unique_coords
			]
			
			# Write line directly
			line = f"{image_name}\t{json.dumps(all_label_dict, ensure_ascii=False)}"
			f.write(line + "\n")

			
							

	   
	  
