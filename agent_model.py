# agent_model.py
import torch
from PIL import Image
import numpy as np
import warnings
import traceback
from torch import nn
import re

from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
        model_dir: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
    ):
        super().__init__()

        # -------- device & dtype ------------------------------------------------
        if device == "mps":
            device = "mps" if (hasattr(torch.backends, "mps") and
                               torch.backends.mps.is_available()) else "cpu"

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.vlm_dtype  = (
            torch.bfloat16 if self.device.type == "cuda" and
                              torch.cuda.is_bf16_supported()
            else torch.float16 if self.device.type != "cpu"
            else torch.float32
        )

        # -------- load model / processor ---------------------------------------
        self.model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            model_dir,
                            torch_dtype = self.vlm_dtype,
                            device_map  = self.device,
                            trust_remote_code = True
                        ).eval()
        self.model.visual.requires_grad_(False)

        self.processor = AutoProcessor.from_pretrained(
                            model_dir, trust_remote_code = True
                         )
        self.tokenizer  = AutoTokenizer.from_pretrained(model_dir,  trust_remote_code=True)

        self.patch_dim = getattr(self.model.config.vision_config, "out_hidden_size", 2048)

        # Ensure projection layer uses the correct dtype AND device
        self.intermediate_state = 256 # SB3 feature size
        self.proj = nn.Linear(self.patch_dim, self.intermediate_state).to(device=self.device, dtype=torch.float32)


    def embed(self, imgs: torch.Tensor, max_batch=None) -> torch.Tensor:
        num_images = imgs.shape[0]
        imgs = imgs.to(self.device)

        try:
            pil_images = [Image.fromarray(x.permute(1,2,0).byte().cpu().numpy()) for x in imgs]
            batch_messages = [
                [{"role": "user", "content": [{"type": "image", "image": img}]}]
                for img in pil_images
            ]

            inputs = self.processor.apply_chat_template(
                batch_messages, padding=True, truncation=True, tokenize=True,
                return_tensors="pt", return_dict=True, add_generation_prompt=False
            ).to(self.device)

            # --- Prepare final inputs ---
            final_inputs = {
                'input_ids': inputs['input_ids'].to(torch.long),
                'pixel_values': inputs['pixel_values'].to(self.vlm_dtype),
                'image_grid_thw': inputs['image_grid_thw'].to(torch.long)
            }
            if 'attention_mask' in inputs:
                final_inputs['attention_mask'] = inputs['attention_mask'].to(torch.long)

            # --- Model Forward Pass ---
            model_output = self.model(**final_inputs, output_hidden_states=True, output_attentions=False)

            # --- Extract Embeddings ---
            # Use layer -8 (index 28 for a 36-layer model)
            extraction_layer_index = -8

            num_layers = self.model.config.num_hidden_layers # Should be 36
            positive_index = extraction_layer_index if extraction_layer_index >= 0 else num_layers + extraction_layer_index
            if not (0 <= positive_index < len(model_output.hidden_states)):
                 raise IndexError(f"Layer index {extraction_layer_index} is out of bounds for hidden_states length {len(model_output.hidden_states)}")
            intermediate_hidden_state = model_output.hidden_states[extraction_layer_index]

            # --- Find Image/Vision Token Index ---
            input_ids = final_inputs['input_ids']
            image_token_id = self.model.config.image_token_id
            vision_token_id = self.model.config.vision_token_id
            batch_indices = torch.arange(num_images, device=self.device)

            token_indices = (input_ids == image_token_id).long().argmax(dim=1)
            if not torch.all(input_ids[batch_indices, token_indices] == image_token_id):
                 token_indices = (input_ids == vision_token_id).long().argmax(dim=1)
                 if not torch.all(input_ids[batch_indices, token_indices] == vision_token_id):
                      raise ValueError(f"Could not reliably find image_token ({image_token_id}) or vision_token ({vision_token_id}) index.")

            extracted_embeddings = intermediate_hidden_state[batch_indices, token_indices]

            return extracted_embeddings

        except Exception as e:
            print(f"[embed ERROR] Failed during embedding: {type(e).__name__}: {e}")
            traceback.print_exc()
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
                inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)

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
            traceback