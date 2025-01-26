from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import os
import csv
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.StreamHandler()])

model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
executor = ThreadPoolExecutor(max_workers=32)


def run_taks_sync(task_prompt, image):
    # sync version of run_example
    return run_task_by_florence_mode(task_prompt, None, image)

async def process_image(image_path):
    logging.info(f"processing image: {image_path}")
    loop = asyncio.get_event_loop()
    pil_image = Image.open(image_path).convert("RGB")
    caption_result = await loop.run_in_executor(executor, run_taks_sync, '<CAPTION>', pil_image)
    detailed_caption_result = await loop.run_in_executor(executor, run_taks_sync, '<DETAILED_CAPTION>', pil_image)
    more_detailed_caption_result = await loop.run_in_executor(executor, run_taks_sync, '<MORE_DETAILED_CAPTION>', pil_image)
    return [image_path, caption_result, detailed_caption_result, more_detailed_caption_result]

def run_task_by_florence_mode(task_prompt, text_input=None,image=None):
    logging.info(f"run task: {task_prompt}")
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input


    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

async def generate_csv_async():
    output_file = '/home/huqianghui/florence_inference-finetune/florence_output.csv'
    rows = []

    logging.info(f"read file from targe directory...")
    for root, dirs, files in os.walk('/home/huqianghui/florence_inference-finetune/人物图'):
        for file in files:
            if file.lower().endswith(('.jpg','.jpeg','.png')):
                image_path = os.path.join(root, file)
                rows.append(image_path)

    # process all images in parallel
    logging.info(f"begin process images...")
    results = await asyncio.gather(*(process_image(image_path) for image_path in rows))

    logging.info(f"write the result to csv file...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "CAPTION", "DETAILED_CAPTION", "MORE_DETAILED_CAPTION"])
        for row in results:
            writer.writerow(row)    

if __name__ == "__main__":
    asyncio.run(generate_csv_async())