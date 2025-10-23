import asyncio

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLM():
    def __init__(self, debug: bool = False) -> None:
        super().__init__()
        # Configurations and Paths
        general_instruction_path = "src/system_prompt.txt"

        with open(general_instruction_path, "r") as f:
            self.general_instruction = f.read().strip()

        self.model_name = "PATH"
        self.debug = debug


        # Loading Stuff
        is_half = True if torch.cuda.is_available() else False
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16 if is_half else torch.float32,
            quantization_config=bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    async def generate(self, prompt: str, speaker: str) -> str:
        """
        Generate text based on the provided prompt.
        """
        messages = [
            {"role": "system", "content": self.general_instruction},
            {"role": "user", "content": f"[{speaker}] {prompt}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.7,
                temperature=0.8,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response
