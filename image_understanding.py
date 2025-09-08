import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(model_path)

model = AutoModelForImageTextToText.from_pretrained(
   model_path,
   torch_dtype=torch.bfloat16,
   _attn_implementation="flash_attention_2"
).to("cuda")

image1_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"

image2_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

messages = [
   {
       "role": "user",
       "content": [
           {"type": "text", "text": "What are the differences between these two images?"},
         {"type": "image", "url": image1_url},
         {"type": "image", "url": image2_url},
       ]
   },
]

inputs = processor.apply_chat_template(
   messages,
   add_generation_prompt=True,
   tokenize=True,
   return_dict=True,
   return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(
   generated_ids,
   skip_special_tokens=True,
)
print(generated_texts[0])

