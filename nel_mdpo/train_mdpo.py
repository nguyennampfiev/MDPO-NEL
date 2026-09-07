import argparse
import glob
import random

from datasets import load_dataset
from trl import DPOConfig
from unsloth import FastLanguageModel

from .trainers import CandidateCollator, MDPOTrainer, MultiNegCollator, MultiNegDPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NEL selector with multi-negative DPO or MDPO.")
    p.add_argument("--train-files", nargs="+", required=True)
    p.add_argument("--eval-files", nargs="+", required=True)
    p.add_argument("--model-name", default="openai/gpt-oss-120b")
    p.add_argument("--output-dir", default="outputs/mdpo")
    p.add_argument("--objective", choices=["multidpo", "mdpo"], default="mdpo")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-answer-length", type=int, default=128)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def expand_files(patterns: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        files.extend(matches or [pattern])
    return files


def normalize_multidpo(example: dict) -> dict:
    rejected = example.get("rejected", [])
    if isinstance(rejected, str):
        rejected = [rejected] if rejected else []
    if len(rejected) > 1:
        rejected = random.sample(rejected, len(rejected))
    example["rejected"] = rejected
    return example


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": expand_files(args.train_files), "eval": expand_files(args.eval_files)},
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    if args.objective == "multidpo":
        train_dataset = train_dataset.map(normalize_multidpo).filter(lambda x: len(x["rejected"]) > 0)
        eval_dataset = eval_dataset.map(normalize_multidpo).filter(lambda x: len(x["rejected"]) > 0)
        trainer_cls = MultiNegDPOTrainer
        collator = MultiNegCollator(tokenizer, args.max_prompt_length, args.max_answer_length)
    else:
        trainer_cls = MDPOTrainer
        collator = CandidateCollator(tokenizer, args.max_prompt_length)

    trainer = trainer_cls(
        model=model,
        ref_model=None,
        data_collator=collator,
        args=DPOConfig(
            bf16=True,
            remove_unused_columns=False,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            output_dir=args.output_dir,
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
