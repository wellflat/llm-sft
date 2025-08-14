import argparse
import typing

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from trl import SFTConfig, SFTTrainer

#if typing.TYPE_CHECKING:
#    from datasets import Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Finetuning with Lora")
    parser.add_argument("--train", "-t", action="store_true", default=True, help="training mode")
    parser.add_argument("--eval", "-e", action="store_true", default=False, help="evaluation mode")
    parser.add_argument("--model", "-m", type=str, default="openai/gpt-oss-20b", help="model name")
    parser.add_argument("--dataset", "-d", type=str, default="HuggingFaceH4/Multilingual-Thinking", help="dataset name")
    parser.add_argument("--push-hub", action="store_true", default=False, help="push to huggingface hub")
    return parser.parse_args()


def prepare_dataset(dataset_name: str) -> Dataset:
    dataset: Dataset = load_dataset(dataset_name, split="train")
    print(type(dataset))
    return dataset


def sample_generate(model, tokenizer) -> None:
    messages = [
        {"role": "user", "content": "¿Cuál es el capital de Australia?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(output_ids)[0]
    print(response)


def train(model, push_hub: bool = False) -> [SFTTrainer, SFTConfig]:
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    training_args = SFTConfig(
        learning_rate=2e-4,
        gradient_checkpointing=True,
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_length=2048,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        output_dir="gpt-oss-20b-multilingual-reasoner",
        report_to="trackio",
        push_to_hub=push_hub,
    )

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    return (trainer, training_args)


if __name__ == "__main__":
    args = parse_args()

    dataset = prepare_dataset(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    messages = dataset[0]["messages"]
    conversation = tokenizer.apply_chat_template(messages, tokenize=False)
    print(conversation)

    quantization_config = Mxfp4Config(dequantize=True)
    print(quantization_config)
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization_config,
        "use_cache": False,  # `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`
        "device_map": "auto",
    }

    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
    print(model)

    trainer, training_args = train(model)

    trainer.save_model(training_args.output_dir)
    trainer.push_to_hub(dataset_name=args.dataset)
