import torch
import os

from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from peft import PeftModel

# If your VideoLlavaForConditionalGeneration class is located elsewhere,
# adjust the import accordingly:
# e.g., from videollava import VideoLlavaForConditionalGeneration
# ^ Adjust this import if you have a custom local definition.

BASE_MODEL_PATH = "LanguageBind/Video-LLaVA-7B-hf"  # Pretrained Video-LLaVA base model
ADAPTER_PATH    = "/home/acapdevi/Video-LLaVA/videollava-7b-minecraft_v2"  # Directory with LoRA adapter files
SAVE_PATH       = "/home/acapdevi/Video-LLaVA/final_finetuned_model"      # Where to save merged final model

print("Loading base model...")
base_model = VideoLlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH
)
print("Base model loaded.")

print("Loading LoRA adapter...")
# Load the LoRA weights onto the base model
model_with_lora = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
print("LoRA adapter loaded.")

print("Merging LoRA weights into base model...")
# Merge the LoRA weights into the base model weights in-place
merged_model = model_with_lora.merge_and_unload()
print("LoRA weights merged. Model is now a standard Hugging Face model.")

print("Loading processor...")
processor = VideoLlavaProcessor.from_pretrained(BASE_MODEL_PATH)
print("Processor loaded.")

print("Saving merged model and processor...")
os.makedirs(SAVE_PATH, exist_ok=True)
merged_model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print(f"Saved the merged model + processor to: {SAVE_PATH}")


