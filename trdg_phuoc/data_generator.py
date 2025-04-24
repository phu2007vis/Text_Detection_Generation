import random
from trdg_phuoc.utils import mask_to_bboxes
from PIL import Image
from trdg_phuoc.generators.wrapper import WrapImage
from trdg_phuoc import computer_text_generator
class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """Same as generate, but takes all parameters as one tuple"""
        return cls.generate(*t)

    @classmethod
    def generate(
        cls,
        text: str,
        font: str,
        size: int,  # image height
        text_color: str,
        orientation: int,
        space_width: int,
        character_spacing: int,
        word_split: bool,
        stroke_width: int,
        stroke_fill,
        wrapper: WrapImage = None,
        num_boxes: int = 3,
        **kwargs
    ) -> tuple[Image, Image, list]:
        # Initialize constants
        MARGINS = {'top': 5, 'left': 5, 'bottom': 5, 'right': 5}
        vertical_margin = MARGINS['top'] + MARGINS['bottom']
        
        # Initialize background
        wrap_index = random.randint(0,len(wrapper)-1)

        background_img = Image.fromarray(wrapper[wrap_index]).convert("RGBA")
        bg_width, bg_height = background_img.size
        background_mask = Image.new("RGB", (bg_width, bg_height), (0, 0, 0))
        
        # Track used positions to prevent overlap
        occupied_areas = []
        
        # Calculate maximum dimensions for text boxes
        max_text_height = size - vertical_margin
        available_height = bg_height - vertical_margin
        available_width = bg_width - (MARGINS['left'] + MARGINS['right'])
        
        # Generate text boxes
        for i in range(min(num_boxes, available_height // max_text_height)):
            # Generate text image
            image, mask = computer_text_generator.generate(
                text=text,
                font=font,
                text_color=text_color,
                font_size=size,
                orientation=orientation,
                space_width=space_width,
                character_spacing=character_spacing,
                word_split=word_split,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
                mask_colr=i+1,
                **kwargs
            )
            
            # Calculate resized dimensions
            aspect_ratio = image.size[0] / image.size[1]
            new_height = min(max_text_height, available_height - (i * max_text_height))
            new_width = min(int(new_height * aspect_ratio), available_width)
            
            if new_width <= 0 or new_height <= 0:
                continue
                
            # Resize images
            resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
            
            # Calculate position (stack vertically with spacing)
            pos_x = MARGINS['left']
            pos_y = MARGINS['top'] + (i * max_text_height)
            
            # Check for overlap and adjust if necessary
            current_box = (pos_x, pos_y, pos_x + new_width, pos_y + new_height)
            while any(cls._boxes_overlap(current_box, occupied) for occupied in occupied_areas):
                pos_y += max_text_height // 2
                current_box = (pos_x, pos_y, pos_x + new_width, pos_y + new_height)
                if pos_y + new_height > bg_height - MARGINS['bottom']:
                    break
                    
            if pos_y + new_height <= bg_height - MARGINS['bottom']:
                # Apply the text to background
                background_img.paste(resized_img, (pos_x, pos_y), resized_img)
                background_mask.paste(resized_mask, (pos_x, pos_y))
                occupied_areas.append(current_box)
        
        # Convert to final format
        final_image = background_img.convert('RGB')
        final_mask = background_mask.convert('RGB')
        
        final_image, final_mask = wrapper.put_image_back(final_image, wrap_index, final_mask)
        
        return final_image, final_mask, mask_to_bboxes(final_mask)
    
    @staticmethod
    def _boxes_overlap(box1: tuple, box2: tuple) -> bool:
        """Check if two boxes overlap"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        return (x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3)
