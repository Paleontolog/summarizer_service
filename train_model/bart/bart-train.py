import argparse

from transformers import (MBartTokenizer,
                          MBartForConditionalGeneration,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)
from transformers.trainer_utils import get_last_checkpoint

from bart.bartdataset import MBartSummarizationDataset
from data_utils import *


def train(args, tokenizer, model, train_records, val_records):
    train_dataset = MBartSummarizationDataset(
        train_records,
        tokenizer,
        args.max_source_tokens_count,
        args.max_target_tokens_count,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )
    val_dataset = MBartSummarizationDataset(
        val_records,
        tokenizer,
        args.max_source_tokens_count,
        args.max_target_tokens_count,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang
    )
    if args.freeze_body:
        for param in model.model.parameters():
            param.requires_grad = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        do_train=True,
        do_eval=True,
        overwrite_output_dir=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_accumulation_steps=args.gradient_accumulation_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        label_smoothing_factor=args.label_smoothing_factor,
        evaluation_strategy="steps"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    checkpoint = get_last_checkpoint(args.last_checkpoint)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/mbart-large-cc25")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/BART")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-source-tokens-count", type=int, default=600)
    parser.add_argument("--max-target-tokens-count", type=int, default=160)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--src-lang", type=str, default="ru_RU")
    parser.add_argument("--tgt-lang", type=str, default="ru_RU")
    parser.add_argument("--label-smoothing-factor", type=float, default=0.1)
    parser.add_argument("--freeze-body", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--fp16-opt-level", type=str, default="O1")

    args = parser.parse_args([])
    tokenizer = MBartTokenizer.from_pretrained(args.model_name, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    model = MBartForConditionalGeneration.from_pretrained(args.model_name).to(args.device)
    train_records = read_gazeta(args.train_file)
    val_records = read_gazeta(args.val_file)
    train(args, model, tokenizer, train_records, val_records)
