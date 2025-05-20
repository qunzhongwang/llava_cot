# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
import os


warnings.filterwarnings("ignore")
pretrained = "CodeGoat24/UnifiedReward-Think-7b"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
cache_dir = "/m2v_intern/wangqunzhong/llava/unifiedRewardThink"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, cache_dir, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

url = "https://github.com/LLaVA-VL/blog/blob/main/2024-10-03-llava-critic/static/images/critic_img_seven.png?raw=True"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
Query = 'What does this image present?'
R1 = 'The image is a black and white sketch of a line that appears to be in the shape of a cross. The line is a simple and straightforward representation of the cross shape, with two straight lines intersecting at a point.'
R2 = 'This is a handwritten number seven.'

question = ("<image>\nGiven a question and a reference image, please analyze in detail the two provided answers (Answer 1 and Answer 2). " \
            "Evaluate them based on the following three core dimensions:\n" \
            "1. Semantic accuracy: How well the answer reflects the visual content of the image\n" \
            "2. Correctness: Whether the answer is logically and factually correct\n" \
            "3. Clarity: Whether the answer is clearly and fluently expressed\n" \
            "You may also consider additional dimensions if you find them relevant (e.g., reasoning ability, attention to detail, multimodal grounding, etc.). " \
            "For each dimension, provide a score from 1 to 10 for both answers, and briefly explain your reasoning. " \
            "Then, compute the total score for each answer by explicitly adding the scores for all dimensions and showing the full calculation. " \
            "Enclose your full reasoning within <think> and </think> tags. " \
            "Then, in the <answer> tag, output exactly one of the following: 'Answer 1 is better' or 'Answer 2 is better'. No other text is allowed in the <answer> section.\n\n" \
            "Example format:\n" \
            "<think>\n" \
            "1. Semantic accuracy: Answer 1 (9/10) - ...; Answer 2 (7/10) - ...\n" \
            "2. Correctness: Answer 1 (8/10) - ...; Answer 2 (7/10) - ...\n" \
            "3. Clarity: Answer 1 (9/10) - ...; Answer 2 (8/10) - ...\n" \
            "[Additional dimensions if any]: Answer 1 (6/10) - ...; Answer 2 (7/10) - ...\n" \
            "Total score:\nAnswer 1: 9+8+9+6=32\nAnswer 2: 7+7+8+7=29\n" \
            "</think>\n" \
            "<answer>Answer 1 is better</answer>\n\n" \
            "**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given answers.**\n\n"
            f"Your task is provided as follows:\nQuestion: [{Query}]\nAnswer 1: [{R1}]\nAnswer 2: [{R2}]")
            
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
