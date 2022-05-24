import argparse

from pytorch_transformers import BertForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainingArguments, Trainer, BertTokenizer
from transformers.integrations import TensorBoardCallback

from create_dataset import BinaryClassificationDataset


class PrinterCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


def train(args):
    writer = SummaryWriter('/content/drive/MyDrive/BINARY_CLASSIFICATION_BERT')
    train_data = BinaryClassificationDataset(train_encodings, train_labels)

    test_data = BinaryClassificationDataset(test_encodings, test_labels)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    model = BertForSequenceClassification.from_pretrained(args.last_checkpoint, cache_dir=args.cache_dir, num_labels=2)
    model.to(args.device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        do_train=True,
        do_eval=True,
        do_predict=False,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        # compute_metrics=compute_metrics,
        callbacks=[PrinterCallback]
    )

    trainer.add_callback(TensorBoardCallback(writer))

    train_result = trainer.train(resume_from_checkpoint=args.last_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str,
                        default="/checkpoint-5500")

    parser.add_argument("--train-file", type=str, default="/train")
    parser.add_argument("--val-file", type=str, default="/eval")
    parser.add_argument("--test-file", type=str, default="/predict")
    parser.add_argument("--cache-dir", type=str, default="/content/cache")
    parser.add_argument("--output-dir", type=str, default="/BINARY_CLASSIFICATION_BERT_MODEL")
    parser.add_argument("--preprocessing-num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.1)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--fp16-opt-level", type=str, default="O1")
    parser.add_argument("--last-checkpoint", type=str,
                        default="/checkpoint-5500")

    args = parser.parse_args([])

    train(args)