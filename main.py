import os

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from deep_translator import GoogleTranslator

MAX_LENGTH = 20
NUM_BEAMS = 4

translator = GoogleTranslator(source='en', target='ru')
model_path = "model"
model = VisionEncoderDecoderModel.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def generate_caption(_image: Image):
    if _image.mode != 'RGB':
        _image = _image.convert('RGB')

    pixel_values = image_processor(images=_image,
                                   return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values,
                                max_length=MAX_LENGTH,  # максимальная длина вывода
                                num_beams=NUM_BEAMS,  # число лучей для beam search
                                early_stopping=True)

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


images = [Image.open(f"images/{x}") for x in os.listdir("images")]
for image in images:
    generated_caption = generate_caption(image)
    translated = translator.translate(generated_caption)
    print(f"Описание картинки [{image.filename}]:", translated)
