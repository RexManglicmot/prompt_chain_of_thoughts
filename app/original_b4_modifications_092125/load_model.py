# app/load_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import cfg

MODEL_ID = cfg.models.primary

# dtype: bf16 on A100/H100, else fp16
major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
DTYPE = torch.bfloat16 if major >= 8 else torch.float16

# Load the tokenizer for the selected Hugging Face model.
# AutoTokenizer picks the correct tokenizer class based on the model’s config.
# from_pretrained(MODEL_ID) will download (or reuse cached) vocab/merges files for that model.
# use_fast=True opts into the Rust-backed `tokenizers` (usually much faster)
# and will silently fall back to the Python tokenizer if a fast version isn’t available.
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,                 # (new) silence torch_dtype deprecation
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ensure a valid pad token
# sometimes this is a hard requirement per LLM
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
try:
    model.generation_config.pad_token_id = tok.pad_token_id
except Exception:
    pass


# Added 9/19
def _wrap_chat(s: str) -> str:
    """Use the model's chat template if available; fall back to raw prompt."""
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return s

@torch.inference_mode()
def call_model(prompt: str, max_new_tokens: int, temperature: float) -> str:
    # Wrap into chat format (helps many instruct LLMs obey sections)
    prompt = _wrap_chat(prompt)

    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(model.device, non_blocking=True) for k, v in enc.items()}
    inp_len = enc["input_ids"].shape[1]
    do_sample = bool(temperature and float(temperature) > 0)

    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,                 # ensure at least 1 token
        do_sample=do_sample,
        temperature=max(1e-5, float(temperature)),
        top_p=0.95 if do_sample else 1.0,
        top_k=50 if do_sample else 0,
        num_beams=1,
        use_cache=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # return ONLY the continuation (no echoed prompt)
    new_tokens = gen[0][inp_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

#END

# def _wrap_chat(s: str) -> str:
#     """Use the model's chat template if available (improves instruction-following)."""
#     try:
#         return tok.apply_chat_template(
#             [{"role": "user", "content": s}],
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#     except Exception:
#         return s

# @torch.inference_mode()
# def call_model(prompt: str, max_new_tokens: int, temperature: float) -> str:
#     """
#     Return ONLY the generated continuation (no prompt echoed).
#     This makes 'Final Answer:' be followed by the model's text, not the prompt.
#     """
#     prompt = _wrap_chat(prompt)

#     enc = tok(prompt, return_tensors="pt")
#     enc = {k: v.to(model.device, non_blocking=True) for k, v in enc.items()}

#     input_len = enc["input_ids"].shape[1]
#     do_sample = temperature and float(temperature) > 0.0

#     gen = model.generate(
#         **enc,
#         max_new_tokens=max_new_tokens,
#         #9/19
#         # Rex added from run_eval and config
#         # max_new_tokens_direct=max_new_tokens_direct,
#         # max_new_cot=max_new_cot,
#         # #END
#         min_new_tokens=1,                 # ensure at least one token is produced
#         do_sample=do_sample,
#         temperature=max(1e-5, float(temperature)),
#         top_p=0.95 if do_sample else 1.0,
#         top_k=50 if do_sample else 0,
#         num_beams=1,
#         use_cache=True,
#         pad_token_id=tok.pad_token_id,
#         eos_token_id=tok.eos_token_id,
#     )

#     # decode ONLY newly generated tokens
#     new_tokens = gen[0][input_len:]
#     text = tok.decode(new_tokens, skip_special_tokens=True).strip()
#     return text



# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from app.config import cfg

# MODEL_ID = cfg["models"].get("primary", "BioMistral/BioMistral-7B")

# # Choose dtype: bfloat16 on A100/H100, otherwise fp16
# if torch.cuda.is_available():
#     major = torch.cuda.get_device_capability(0)[0]
# else:
#     major = 0
# DTYPE = torch.bfloat16 if major >= 8 else torch.float16

# # Load tokenizer and model onto GPU once
# tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=DTYPE,
#     device_map="auto",            # put weights on GPU
#     attn_implementation="sdpa",   # fast PyTorch attention
# )
# model.eval()

# # Speed tweaks
# torch.backends.cuda.matmul.allow_tf32 = True
# try:
#     torch.set_float32_matmul_precision("high")
# except Exception:
#     pass

# # Ensure a valid pad token
# if tok.pad_token_id is None:
#     tok.pad_token_id = tok.eos_token_id
# try:
#     model.generation_config.pad_token_id = tok.pad_token_id
# except Exception:
#     pass

# @torch.inference_mode()
# def call_model(prompt: str, max_new_tokens: int, temperature: float) -> str:
#     """Generate output from the model with minimal CPU overhead."""
#     enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
#     enc = {k: v.to(model.device, non_blocking=True) for k, v in enc.items()}

#     do_sample = temperature and temperature > 0.0
#     gen = model.generate(
#         **enc,
#         max_new_tokens=max_new_tokens,
#         do_sample=do_sample,
#         temperature=max(1e-5, float(temperature)),
#         top_p=0.95 if do_sample else 1.0,
#         top_k=50 if do_sample else 0,
#         num_beams=1,
#         use_cache=True,
#     )
#     return tok.decode(gen[0], skip_special_tokens=True)
