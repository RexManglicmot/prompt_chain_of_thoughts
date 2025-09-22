# app/load_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import cfg

# Obtain model name from config object into another object
# Remember, this is a constant and should not be changed
MODEL_ID = cfg.models.primary

# If CUDA is available, query device 0's compute capability and take the MAJOR version; otherwise fall back to 0 (meaning "no CUDA").
# Check if torch.cuda.is_available() first
    # True -> evaluate torch.cuda.get_device_capability(0)[0]
    # False -> evaluate 0. The other branch is not evaluated.
# torch.cuda.get_device_capability(0)[0]
    # Call get_device_capability(0) → returns a tuple (major, minor) describing the GPU’s CUDA compute capability for device index 0.
    # Index [0] → take the major part of that tuple. THIS is what is wanted.
major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0

# Pick a tensor dtype based on the GPU’s compute capability
# Want bfloat16 if it is available
DTYPE = torch.bfloat16 if major >= 8 else torch.float16

# Load the tokenizer for the selected Hugging Face model.
# AutoTokenizer picks the correct tokenizer class based on the model’s config.
# from_pretrained(MODEL_ID) will download (or reuse cached) vocab/merges files for that model.
# use_fast=True opts into the Rust-backed `tokenizers` (usually much faster)
# and will silently fall back to the Python tokenizer if a fast version isn’t available.
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Load the causal language model weights from Hugging Face (or local cache)
# .from_pretrained() instantiates a model/tokenizer/config from a repo on the Hub or a local folder that contains the saved files.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,                    # BioMistral
    dtype=DTYPE,                 # constant set above
    device_map="auto",           # Let Accelerate/Transformers place layers automatically across available devices. On 1 GPU → the whole model goes to that GPU
    attn_implementation="sdpa",     # Use PyTorch Scaled Dot-Product Attention (SDPA) backend for efficiency/stability
)

# Put model in inference mode
model.eval()

# Allow TensorFloat-32 (TF32) on CUDA matmuls (Ampere/Hopper GPUs).
# TF32 uses 10-bit mantissa for FP32 ops → big speedup with tiny precision loss.
torch.backends.cuda.matmul.allow_tf32 = True

try:
    # PyTorch 2.x hint: choose faster FP32 matmul kernels.
    # On CUDA (with TF32 allowed) this typically engages TF32 paths.
    # Values: "highest" | "high" | "medium" (default often "high"/"medium").
    torch.set_float32_matmul_precision("high")
except Exception:
    # Older PyTorch or non-CUDA backends may not support this; safe to ignore.
    pass

# Padding Safety for generation
# Sometimes this is a hard requirement per LLM
# Some decoder-only LMs *require* a pad token for batching/masking.
# A pad token is a special placeholder token used to make all sequences in a batch the same length so they can be processed together efficiently.
# If the tokenizer has no pad token, reuse EOS as PAD (common practice).
# tok.pad_token_id  and tok.eos_token_id comes from the tokenizer tokenizer files on Hugging Face
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
try:
    # Keep model's generation config in sync so generate() doesn't warn/error.
    model.generation_config.pad_token_id = tok.pad_token_id
except Exception:
    # Some models don’t expose/allow mutable generation_config; ignore if so.
    pass

# Format a plain user string `s` into the model's chat prompt using the tokenizer's built-in chat template.
def _wrap_chat(s: str) -> str:
    try:
        # apply_chat_template(...) injects the right special tokens and role tags 
        # (e.g., system/user/assistant, BOS/EOS) for models like Llama, Qwen, Mistral chat, etc.
        return tok.apply_chat_template(
            # Minimal single-turn chat: one user message
            [{"role": "user", "content": s}],
            # return a STRING, not token IDs
            tokenize=False,
            # append the "assistant" header/tag so the model knows to continue as the assistant
            add_generation_prompt=True,
        )
    except Exception:
        # If the tokenizer has no chat_template or raises (e.g., non-chat model),
        # just use the raw text; generation will still work with plain prompts.
        return s

# disable autograd for faster, lower-memory inference
@torch.inference_mode()
def call_model(prompt: str, 
               max_new_tokens: int, 
               temperature: float) -> str:
    
    # Format the plain text into the model's expected chat template (if available)
    # Use function above
    prompt = _wrap_chat(prompt)

    # Tokenize the prompt into model-ready PyTorch tensors
    # return_tensors="pt" makes the outputs torch.Tensors (batch size = 1).
    # `enc` is a dict like {"input_ids": LongTensor[[...]], "attention_mask": LongTensor[[...]]}
    enc = tok(prompt, return_tensors="pt")

    # Move tensors onto the same device as the model (GPU/CPU). 
    # non_blocking=True can help with pinned CPU memory.
    enc = {k: v.to(model.device, non_blocking=True) for k, v in enc.items()}

    # Record the number of input (prompt) tokens so we can remove them from the
    # generated sequence later. `enc["input_ids"]` has shape [batch_size, seq_len];
    # here batch_size == 1, so shape[1] is the prompt length in tokens.
    inp_len = enc["input_ids"].shape[1]

    # Sampling is enabled only if temperature > 0; otherwise greedy decoding
    # Sampling means we draw the next token at random from that distribution 
    # (often after tweaks like temperature, top-k, top-p).
    do_sample = bool(temperature and float(temperature) > 0)

    # Generate a continuation from the model
    gen = model.generate(
        **enc,                            # unpack tokenizer outputs: input_ids, attention_mask as PyTorch tensor

        # Length controls
        max_new_tokens=max_new_tokens,    # hard cap on *newly generated* tokens (excludes the prompt)
        min_new_tokens=1,                 # ensure at least 1 token

        # Decode
        do_sample=do_sample,              # True → stochastic sampling; False → greedy (argmax) decoding
        temperature=max(1e-5, float(temperature)),
        top_p=0.95 if do_sample else 1.0,
        top_k=50 if do_sample else 0,
        num_beams=1,

        # Speed and Memory
        use_cache=True,                   # reuse KV cache for faster autoregressive

        # Special tokens (padding & stopping)
        pad_token_id=tok.pad_token_id,          # needed for batching/attention masks during generation
        eos_token_id=tok.eos_token_id,          # stop token; generation halts when this is emitted
    )

    # Return ONLY the model’s continuation, not the echoed prompt:
    # `gen` is a tensor of token IDs with shape [batch, total_len]
    # It contains the full sequence = [prompt tokens ... generated tokens]
    # We slice off the first `inp_len` tokens (the prompt) to keep only new tokens
    new_tokens = gen[0][inp_len:]

    # Convert token IDs INTO text:
    # skip_special_tokens=True removes things like <eos>, <pad>, special chat tags, etc.
    # .strip() trims leading/trailing whitespace/newlines
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


