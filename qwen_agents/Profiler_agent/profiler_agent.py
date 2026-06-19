from huggingface_hub import login
#from google.colab import userdata
import torch
import os
from pydantic_ai import Agent
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

Profiler_SKILL = "qwen_agents\Profiler_agent\.agents\skills\criminal-behavioral-analysis\SKILL.md"

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Kizzington/Qwen3-VL-8B-Thinking-heretic", dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Thinking",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

image_path = "/content/image.png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at: {image_path}. Please ensure the image is uploaded or the path is correct.")

messages = [
    {
        "role": "system",
        "content": "You are a criminal profiler.\n\n"
        f"BEFORE doing anything else, call read_file('{Profiler_SKILL}') "
        "to load your operating instructions, then follow every step exactly.\n\n"
        "MCP servers available:\n"
        "fetch_image_base64, delegate_image_to_synthesis, "
        "delegate_to_expert_agent, store_result, utc_now, log_event"

    },

    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
