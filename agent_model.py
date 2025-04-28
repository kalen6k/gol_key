# agent_model.py
import torch
from PIL import Image
import numpy as np
import warnings
import traceback
from torch import nn
import re
import time

from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

PROMPT = (
    # ---- task description (unchanged) -----------------------------------
    "You are playing **GOL-Key**. Some cells become immortal once they extend "
    "the neighbouring prefix of a hidden target string. A valid prefix forms "
    "an edge-adjacent path that may run horizontally, vertically, or zig-zag "
    "in right angles; each step continues the target sequence. Multiple "
    "prefixes can grow and even conflict, but there is only one underlying "
    "string.\n\n"
    # ---- explicit instruction & schema -----------------------------------
    "## Instructions\n"
    "1. Look at the board and infer the **entire hidden string**.\n"
    "2. **Respond with exactly one line** in the format:\n"
    "`guess:<string>`\n"
    "   – use lower-case letters only\n"
    "   – no extra words, no punctuation besides the colon\n"
    "3. Do **not** explain your reasoning.\n\n"
)

class GOLKeyAgent:
    def __init__(
        self,
        # === Default to BASE model ===
        model_dir: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        super().__init__()

        # --- Determine Device ---
        resolved_device = "cpu"
        can_use_cuda = torch.cuda.is_available()
        can_use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if can_use_cuda:
            resolved_device = "cuda"
        elif can_use_mps:
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        print(f"GOLKeyAgent: Using device: {self.device}")

        # --- Determine Compute Dtype ---
        if self.device.type == "cuda":
            self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.device.type == "mps":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.float32
        print(f"GOLKeyAgent: Using compute dtype: {self.compute_dtype}")

        # --- Load Standard HF Model ---
        print(f"GOLKeyAgent: Loading standard HF model: {model_dir}")
        try:
            attn_implementation = "sdpa"
            if self.device.type == 'cuda':
                 try:
                      import flash_attn
                      attn_implementation="flash_attention_2"
                      print("GOLKeyAgent: flash_attn package found, requesting 'flash_attention_2'.")
                 except ImportError:
                      print("GOLKeyAgent: flash_attn package not found, using default 'sdpa' attn_implementation.")

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                               model_dir,
                               torch_dtype=self.compute_dtype,
                               attn_implementation=attn_implementation,
                               device_map=None,
                               trust_remote_code=True
                           ).to(self.device).eval()
            self.model.visual.requires_grad_(False)
            print("GOLKeyAgent: Standard HF model loaded successfully.")

            self.patch_dim = getattr(self.model.config.vision_config, "out_hidden_size", 2048)

        except Exception as e:
            print(f"FATAL: Failed to load base model {model_dir}: {e}")
            traceback.print_exc()
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            print("GOLKeyAgent: Tokenizer loaded.")
        except Exception as e:
             print(f"FATAL: Failed to load tokenizer for {model_dir}: {e}")
             traceback.print_exc()
             raise

        try:
            print("GOLKeyAgent: Attempting to load Fast Processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_dir,
                use_fast=True,
                trust_remote_code=True
            )
            print("GOLKeyAgent: Fast Processor loaded successfully.")
        except Exception as e_fast:
            print(f"GOLKeyAgent: Could not load Fast Processor ({e_fast}), falling back to Slow Processor.")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_dir,
                    use_fast=False,
                    trust_remote_code=True
                )
                print("GOLKeyAgent: Slow Processor loaded successfully.")
            except Exception as e_slow:
                 print(f"FATAL: Failed to load even Slow Processor for {model_dir}: {e_slow}")
                 traceback.print_exc()
                 raise

        # --- Initialize Projection Layer ---
        self.intermediate_state = 256
        self.proj = nn.Linear(self.patch_dim, self.intermediate_state).to(device=self.device, dtype=torch.float32)
        self.proj.requires_grad_(True)
        print(f"GOLKeyAgent: Projection layer initialized (dtype: {self.proj.weight.dtype}).")

    def embed(self, imgs: torch.Tensor, max_batch: int | None = None) -> torch.Tensor:
        num_images = imgs.shape[0]
        imgs = imgs.to(self.device)

        if max_batch is None:
            max_batch = num_images

        all_extracted_embeddings = []

        try:
            for i in range(0, num_images, max_batch):
                chunk_imgs = imgs[i : i + max_batch]
                chunk_num_images = chunk_imgs.shape[0]

                #print(f"        GOLKeyAgent: Processing embed chunk {i//max_batch + 1}, size {chunk_num_images}...")
                chunk_start_time = time.time()

                # --- Process this chunk ---
                chunk_pil_images = [Image.fromarray(x.permute(1,2,0).byte().cpu().numpy()) for x in chunk_imgs]
                chunk_batch_messages = [
                    [{"role": "user", "content": [{"type": "image", "image": img}]}]
                    for img in chunk_pil_images
                ]

                chunk_inputs = self.processor.apply_chat_template(
                    chunk_batch_messages, padding=True, truncation=True, tokenize=True,
                    return_tensors="pt", return_dict=True, add_generation_prompt=False
                ).to(self.device)

                chunk_final_inputs = {
                    'input_ids': chunk_inputs['input_ids'].to(torch.long),
                    'pixel_values': chunk_inputs['pixel_values'].to(self.compute_dtype),
                    'image_grid_thw': chunk_inputs['image_grid_thw'].to(torch.long)
                }
                if 'attention_mask' in chunk_inputs:
                    chunk_final_inputs['attention_mask'] = chunk_inputs['attention_mask'].to(torch.long)

                # --- Model Forward Pass ---
                chunk_model_output = self.model(**chunk_final_inputs, output_hidden_states=True, output_attentions=False)
                if not torch.isfinite(chunk_model_output.hidden_states[-1]).all():
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"WARNING: NaNs or Infs detected in model output for chunk {i//max_batch + 1}!")
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                # --- Extract Embeddings (from the chunk output) ---
                extraction_layer_index = -8
                num_layers = self.model.config.num_hidden_layers
                positive_index = extraction_layer_index if extraction_layer_index >= 0 else num_layers + extraction_layer_index
                # Check bounds omitted for brevity, keep it in your code
                chunk_intermediate_hidden_state = chunk_model_output.hidden_states[positive_index]

                chunk_input_ids = chunk_final_inputs['input_ids']
                image_token_id = self.model.config.image_token_id
                vision_token_id = self.model.config.vision_token_id
                chunk_batch_indices = torch.arange(chunk_num_images, device=self.device)

                # Find token indices within the chunk
                chunk_token_indices = (chunk_input_ids == image_token_id).long().argmax(dim=1)
                if not torch.all(chunk_input_ids[chunk_batch_indices, chunk_token_indices] == image_token_id):
                    chunk_token_indices = (chunk_input_ids == vision_token_id).long().argmax(dim=1)

                chunk_extracted_embeddings = chunk_intermediate_hidden_state[chunk_batch_indices, chunk_token_indices]

                all_extracted_embeddings.append(chunk_extracted_embeddings)

                #print(f"        GOLKeyAgent: Finished embed chunk {i//max_batch + 1} (duration: {time.time() - chunk_start_time:.3f}s)")


            final_extracted_embeddings = torch.cat(all_extracted_embeddings, dim=0)

            return final_extracted_embeddings

        except Exception as e:
            print(f"[embed ERROR] Failed during embedding: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Return shape is patch_dim now
            return torch.zeros((num_images, self.patch_dim), device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def guess_word(self, image_tensor: torch.Tensor, target_len) -> str:
        """ Uses the VLM to guess the target word based on the image tensor. """
        if image_tensor.dim() != 3:
            raise ValueError(f"Expected single image tensor with shape (C, H, W), got {image_tensor.shape}")

        image_tensor_permuted = image_tensor.float().permute(1, 2, 0) # H, W, C
        if image_tensor_permuted.max() <= 1.0 and image_tensor_permuted.min() >= 0.0:
            image_tensor_permuted = (image_tensor_permuted * 255)
        pil_image = Image.fromarray(image_tensor_permuted.cpu().numpy().astype(np.uint8))

        dynamic_prompt = f"{PROMPT} The target string has {target_len} letters."

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": dynamic_prompt}
            ]}
        ]

        try:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            images_to_process = [pil_image]

            inputs = self.processor(
                text=prompt_text,
                images=images_to_process,
                return_tensors="pt"
            ).to(self.device)

            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(self.compute_dtype)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            input_ids_len = inputs['input_ids'].shape[1]
            generated_tokens = generated_ids[:, input_ids_len:]
            response = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            raw = response.lower().strip()
            cleaned = ''.join(filter(str.isalpha, response)).lower().strip()
            raw = response.lower().strip()

            m = re.search(r'guess\s*[:=\-]\s*([a-z]+)', raw)
            cleaned = m.group(1) if m else ''

            #print("model:", raw, "→ cleaned:", cleaned)
            return cleaned

        except Exception as e:
            warnings.warn(f"[guess_word ERROR] Failed: {e}. Returning empty string.")
            traceback.print_exc()
            return ""