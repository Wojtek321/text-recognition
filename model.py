from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import get_peft_model, LoraConfig
import torch


model_name = "Salesforce/blip-image-captioning-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
czytacz = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

for name, module in czytacz.named_modules():
    module.requires_grad = False


lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["qkv", "projection", "query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
)

czytacz = get_peft_model(czytacz, lora_config)
czytacz.print_trainable_parameters()
