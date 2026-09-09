from huggingface_hub import login
#from google.colab import userdata
import torch
import os
from typing import Optional
from pydantic_ai import Agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

load_dotenv()

Profiler_SKILL = "qwen_agents\Profiler_agent\.agents\skills\criminal-behavioral-analysis\SKILL.md"

MCP_PORT = int(os.getenv("MCP_SERVER_PORT", "9000"))
 
# Vision tools + A2A delegation — connects to FastMCP SSE server
mcp_toolset = MCPServerSSE(url=f"http://localhost:{MCP_PORT}/sse")
 
# Standard filesystem server — read_file, list_directory, get_file_info
# Sandboxed to SKILLS_DIR so agents can only read their own skill files.
# Requires Node.js >= 18 for npx.
filesystem_toolset = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", str(SKILLS_DIR)],
    env={**os.environ},
)

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

def _make_model(
    model_id: str,
    *,
    adapter_path: Optional[str] = None,
    processor_path: Optional[str] = None,
    temperature: float = 0.1,
):

class BoundingBox(BaseModel):
    label:      str
    confidence: float = Field(ge=0.0, le=1.0)
    position:   str   = Field(description="Rough position in frame: top-left, center …")
 
 
class ProfileAnalysis(BaseModel):
    """Output of Agent 1 — Profile Analyzer."""
    task_id:           str
    scene_description: str
    objects:           list[BoundingBox]
    text_content:      Optional[str] = None
    dominant_colors:   list[str]
    context_category:  str = Field(
        description="nature | urban | product | document | person | other"
    )
    mood:              str
    confidence:        float = Field(ge=0.0, le=1.0)
    timestamp:         str
    forensic_result:  Optional[str] = Field(
        None,
        description="ForensicResult JSON if Agent 1 auto-delegated to Agent 2"
    )

profile_agent: Agent[None, ProfileAnalysis] = Agent(
    model=_make_model(_VISUAL_MODEL_ID),
    output_type=ProfileAnalysis,                  # was result_type= in older pydantic-ai
    system_prompt=(
        "You are a Qwen2-VL visual analyst agent.\n\n"
        f"BEFORE doing anything else, call read_file('{Profiler_SKILL}') "
        "to load your operating instructions, then follow every step exactly.\n\n"
        "Toolsets available:\n"
        "  • Filesystem — read_file, list_directory, get_file_info\n"
        "  • Vision-tools — fetch_image_base64, delegate_image_to_synthesis, "
        "delegate_to_expert_agent, store_result, utc_now, log_event"
    ),
    toolsets=[mcp_toolset, filesystem_toolset],  # was mcp_servers= in older pydantic-ai
    retries=2,
)