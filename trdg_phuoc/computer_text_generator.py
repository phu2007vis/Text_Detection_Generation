import random as rnd
from typing import Tuple
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
import random
from trdg_phuoc.utils import get_text_width, get_text_height

# Thai Unicode reference: https://jrgraphix.net/r/Unicode/0E00-0E7F
TH_TONE_MARKS = [
    "0xe47",
    "0xe48",
    "0xe49",
    "0xe4a",
    "0xe4b",
    "0xe4c",
    "0xe4d",
    "0xe4e",
]
TH_UNDER_VOWELS = ["0xe38", "0xe39", "\0xe3A"]
TH_UPPER_VOWELS = ["0xe31", "0xe34", "0xe35", "0xe36", "0xe37"]


def generate(
    **kwargs
    
) -> Tuple:
    
   
    return _generate_horizontal_text(
        **kwargs,
        
    )


def _compute_character_width(image_font: ImageFont, character: str) -> int:
    if len(character) == 1 and (
        "{0:#x}".format(ord(character))
        in TH_TONE_MARKS + TH_UNDER_VOWELS + TH_UNDER_VOWELS + TH_UPPER_VOWELS
    ):
        return 0
    # Casting as int to preserve the old behavior
    return round(image_font.getlength(character))


def _generate_horizontal_text(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    character_spacing: int = 0 ,
    word_split: bool = False,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
    mask_value= 1,
    
) -> Tuple:
    ratio_scale = 0.04
    num_font_size_variations = len(text)
    
    min_font_size = int((1-ratio_scale) * font_size)
    max_font_size = int((1+ratio_scale) * font_size)
    all_font_size = [
        random.randint(min_font_size,max_font_size) for _ in range(num_font_size_variations)
    ]
    
    all_image_font = [ImageFont.truetype(font=font, size=font_size) for font_size in all_font_size]
   
    all_space_width = [int(get_text_width(image_font, " ") * random.uniform(0.05,0.4)) for image_font in all_image_font]
  
    splitted_text = text

    piece_widths = [
        _compute_character_width(all_image_font[i], p) if p != " " else random.choice(all_space_width)
        for i,p in enumerate(splitted_text)
    ]
    
    text_width = sum(piece_widths)
    if not word_split:
        text_width += character_spacing * (len(text) - 1)

    text_height = max([get_text_height(all_image_font[i], p) for i,p in enumerate(splitted_text)])
  
    background_color = (255,255,255,0)
    txt_img = Image.new("RGBA", (text_width, text_height), background_color)
  
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )

    for i, p in enumerate(splitted_text):
        
        txt_img_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=fill,
            font=all_image_font[i],
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
     
        txt_mask_draw.text(
            (sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0),
            p,
            fill=(mask_value,mask_value,mask_value),
            font=all_image_font[i],
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    return txt_img, txt_mask


def _generate_vertical_text(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
) -> Tuple:
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(get_text_height(image_font, " ") * space_width)

    char_heights = [
        get_text_height(image_font, c) if c != " " else space_height for c in text
    ]
    text_width = max([get_text_width(image_font, c) for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        txt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    
    return txt_img, txt_mask
