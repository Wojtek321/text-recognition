import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftConfig, PeftModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_processor(model_path):
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = BlipForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)

    model = PeftModel.from_pretrained(base_model, model_path).to(device)
    processor = BlipProcessor.from_pretrained(model_path, use_fast=True)

    return model, processor


def predict_text_from_image(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")

    model.to(device)
    model.eval()

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs)

    predicted_text = processor.decode(out[0], skip_special_tokens=True)

    return image, predicted_text


def display_result(image, predicted_text):
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted text: {predicted_text}")
    plt.tight_layout()
    plt.show()


def process_images_in_folder(model_path, folder_path):
    model, processor = load_model_and_processor(model_path)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image, predicted_text = predict_text_from_image(image_path, model, processor)

            print(f"Predicted text for {image_name}: {predicted_text}")
            display_result(image, predicted_text)


def main():
    # model_path = "czytacz/checkpoint-3000"
    model_path = "chincyk/czytacz"
    folder_path = "test"

    process_images_in_folder(model_path, folder_path)


if __name__ == "__main__":
    main()
