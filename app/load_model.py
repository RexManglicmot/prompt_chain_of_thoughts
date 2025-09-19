import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import cfg

MODEL_ID = cfg["models"].get("primary", "BioMistral/BioMistral-7B")

# Choose dtype: bfloat16 on A100/H100, otherwise fp16
if torch.cuda.is_available():
    major = torch.cuda.get_device_capability(0)[0]
else:
    major = 0
DTYPE = torch.bfloat16 if major >= 8 else torch.float16

# Load tokenizer and model onto GPU once
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",            # put weights on GPU
    attn_implementation="sdpa",   # fast PyTorch attention
)
model.eval()

# Speed tweaks
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Ensure a valid pad token
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
try:
    model.generation_config.pad_token_id = tok.pad_token_id
except Exception:
    pass

@torch.inference_mode()
def call_model(prompt: str, max_new_tokens: int, temperature: float) -> str:
    """Generate output from the model with minimal CPU overhead."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(model.device, non_blocking=True) for k, v in enc.items()}

    do_sample = temperature and temperature > 0.0
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(1e-5, float(temperature)),
        top_p=0.95 if do_sample else 1.0,
        top_k=50 if do_sample else 0,
        num_beams=1,
        use_cache=True,
    )
    return tok.decode(gen[0], skip_special_tokens=True)
