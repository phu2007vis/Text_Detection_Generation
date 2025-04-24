from trdg_phuoc.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromStrings,
    GeneratorFromWikipedia,
)

# The generators use the same arguments as the CLI, only as parameters
generator = GeneratorFromStrings(
    ['Test1', 'Test2', 'Test3'],
    blur=2,
    random_blur=True,
    # is_handwritten = True
    fonts=['/work/21013187/phuoc/TextRecognitionDataGenerator/phuoc_fonts/AS Melanie Handwritting.ttf']
)

for img, lbl in generator:
    # Do something with the pillow images here.
    img.save("test.jpg")
    exit()