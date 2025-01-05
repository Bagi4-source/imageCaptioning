import threading
from tkinter import Tk, Button, Label, Text, filedialog

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image, ImageTk
from deep_translator import GoogleTranslator

MAX_LENGTH = 20
NUM_BEAMS = 4


class ImageCaptioning:
    model_path = "model"

    def __init__(self):
        # Initialize Tkinter root
        self.root = Tk()
        self.root.title("Генератор описаний изображений")
        self.root.geometry("400x450")

        # Create UI elements
        self.select_button = Button(self.root, text="Выберите изображение", command=self.__select_image)
        self.select_button.config(state="disabled")
        self.select_button.pack(pady=10)

        self.image_label = Label(self.root)
        self.image_label.pack(pady=10)

        self.caption_text = Text(self.root, height=5, width=50)
        self.caption_text.config(state="disabled", font=("Arial", 16, "bold"))
        self.caption_text.pack(pady=10)

        # Schedule model initialization after the window appears
        self.root.after(100, self.__start_model_initialization)

        # Run the Tkinter event loop
        self.root.mainloop()

    def __start_model_initialization(self):
        # Start model initialization in a separate thread
        threading.Thread(target=self.__init_model).start()

    def __set_caption(self, text: str):
        self.caption_text.configure(state="normal")
        self.caption_text.delete(1.0, "end")
        self.caption_text.insert("end", text)
        self.caption_text.configure(state="disabled")

    def __init_model(self):
        # Show a message indicating that models are being loaded
        self.__set_caption("Инициализация...")

        # Initialize models
        self.translator = GoogleTranslator(source='en', target='ru')
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
        self.image_processor = ViTImageProcessor.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Enable the select button after initialization
        self.select_button.config(state='normal')

        # Clear the message after initialization
        self.__set_caption("Инициализация завершена. Выберите изображение.")

    def generate_caption(self, _image: Image):
        if _image.mode != 'RGB':
            _image = _image.convert('RGB')

        pixel_values = self.image_processor(images=_image,
                                            return_tensors="pt").pixel_values
        output_ids = self.model.generate(pixel_values,
                                         max_length=MAX_LENGTH,  # максимальная длина вывода
                                         num_beams=NUM_BEAMS,  # число лучей для beam search
                                         early_stopping=True)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def __select_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Image Files", "*.jpeg *.jpg *.png *.webp"),
                ("All Files", "*.*")
            ],
        )
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((400, 400))  # Resize for preview
            img_display = ImageTk.PhotoImage(image)

            self.image_label.config(image=img_display)
            self.image_label.image = img_display

            # Show loading indicator
            self.__set_caption("Загрузка...")

            # Process the image and update the caption
            self.root.after(100, self.process_image, image)

    def process_image(self, image):
        generated_caption = self.generate_caption(image)
        translated_caption = self.translator.translate(generated_caption)
        self.__set_caption(translated_caption)


ImageCaptioning()
