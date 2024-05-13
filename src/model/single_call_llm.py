import contextlib
import torch
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100

torch.cuda.empty_cache()


class SingleCallLLM(torch.nn.Module):

    def __init__(
            self,
            args,
            **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            # "max_memory": {0: '20GiB', 1: '80GiB'},
            "max_memory": {0: '24GiB'},
            "device_map": "cuda:0",
            "revision": "main",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)

            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        # Use for local RTX 4090
        model.config.attn_implementation = "flash_attention_2"
        self.model = model
        print('Finish loading LLAMA!')

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        # print(list(self.parameters())[0].device)
        return list(self.parameters())[0].device
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def inference(self, sample):

        # encode description and question
        question = self.tokenizer(sample["question"], add_special_tokens=False)
        description = self.tokenizer(sample["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').to(self.device).input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)

        # Add bos & eos token
        input_ids = description.input_ids[:self.max_txt_len] + question.input_ids + eos_user_tokens.input_ids
        inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

        # pad inputs_embeds
        max_length = inputs_embeds.shape[0]
        pad_length = max_length - inputs_embeds.shape[0]

        inputs_embeds = torch.cat([pad_embeds.repeat(pad_length, 1), inputs_embeds])
        attention_mask = [0] * pad_length + [1] * inputs_embeds.shape[0]

        inputs_embeds = inputs_embeds.to(self.model.device)
        attention_mask = torch.tensor(attention_mask).to(self.model.device)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds.unsqueeze(0),
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask.unsqueeze(0),
                # do_sample=True,
                use_cache=True  # IMPORTANT!
            )

        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            'id': sample['id'],
            'pred': pred,
            'label': sample['label'],
            'question': sample['question'],
            'desc': sample['desc'],
        }
