import argparse
import os
import time
from datetime import datetime

import tqdm
from pytorch_transformers.optimization import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tnrange
from transformers import GPT2LMHeadModel, AdamW

from gpt_3.dataset import GPT21024Dataset, read_gazeta
from gpt_3.utils import add_special_tokens, set_seed


def train(args, model, tokenizer, train_dataset, valid_dataset, ignore_index, load_path=None):
    writer = SummaryWriter('/content/drive/MyDrive/GPT_3')
    train_sampler = RandomSampler(train_dataset)
    train_dl = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                          num_workers=args.num_workers)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)  # ignores padding token for loss calculation

    optimizer = AdamW(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(torch.load(load_path + "/" + 'optimizer.pt'))

    scheduler = WarmupLinearSchedule(optimizer, 100, 80000)
    scheduler.load_state_dict(torch.load(load_path + "/" + 'scheduler.pt'))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = tnrange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm.notebook.tqdm(train_dl, desc="Training")
        for step, batch in enumerate(epoch_iterator):
            inputs = batch['article'].to(args.device)
            model.train()
            logits = model(inputs).logits
            shift_logits = logits[..., batch['sum_idx']:-1, :].contiguous()
            shift_labels = inputs[..., batch['sum_idx'] + 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('loss', (tr_loss - logging_loss) / args.gradient_accumulation_steps, global_step)
                print("loss:", (tr_loss - logging_loss), end='\n\n')
                logging_loss = tr_loss

            if (step + 1) % (args.eval_steps * args.gradient_accumulation_steps) == 0:
                results = evaluate(args, model, valid_dataset, ignore_index, global_step)
                for key, value in results.items():
                    writer.add_scalar('eval_{}'.format(key), value, global_step)
                print('After', global_step + 1, 'updates: ', end='\n\n')

            if (step + 1) % (args.save_steps * args.gradient_accumulation_steps) == 0:
                checkpoint_prefix = "checkpoint"
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)


def evaluate(args, model, eval_dataset, ignore_index, global_step=None):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    eval_output_dir = args.output_dir

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    loss_fct = CrossEntropyLoss(ignore_index=ignore_index)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm.notebook.tqdm(eval_dataloader, desc="Evaluating"):
        inputs = batch['article'].to(args.device)
        labels = inputs
        with torch.no_grad():
            logits = model(inputs).logits
            shift_logits = logits[..., batch['sum_idx']:-1, :].contiguous()
            shift_labels = labels[..., batch['sum_idx'] + 1:].contiguous()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }
    print("perplexity:", perplexity.item())

    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--seed", default=42, type=int, help="seed to replicate results")
    parser.add_argument("--n_gpu", default=1, type=int, help="no of gpu available")
    parser.add_argument("--gradient_accumulation_steps", default=32, type=int, help="gradient_accumulation_steps")
    parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--num_workers", default=4, type=int, help="num of cpus available")
    parser.add_argument("--device", default=torch.device('cuda'), help="torch.device object")
    parser.add_argument("--num_train_epochs", default=2, type=int, help="no of epochs of training")
    parser.add_argument("--output_dir", default='/content/drive/MyDrive/GPT_3/output', type=str,
                        help="path to save evaluation results")
    parser.add_argument("--model_dir", default="sberbank-ai/rugpt3small_based_on_gpt2", type=str,
                        help="path to save trained model")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max gradient norm.")
    parser.add_argument("--root_dir", default='/content/Generating_Text_Summary_With_GPT2/CNN/gpt2_1024_data', type=str,
                        help="location of json dataset.")
    parser.add_argument("--ids_file", default='/content/Generating_Text_Summary_With_GPT2/CNN/ids.json', type=str,
                        help="location of train, valid and test file indexes")
    parser.add_argument("--save-steps", type=int, default=150)
    parser.add_argument("--eval-steps", type=int, default=50)
    args = parser.parse_args()
    print(args)

    train_records = read_gazeta(
        r"D:\PycharmProjects\Summarizer_transformer\datasets\gazeta\prepared\train_records.json")
    val_records = read_gazeta(r"D:\PycharmProjects\Summarizer_transformer\datasets\gazeta\prepared\val_records.json")

    train_data = GPT21024Dataset(train_records)
    valid_data = GPT21024Dataset(val_records)

    tokenizer = add_special_tokens()
    ignore_idx = tokenizer.pad_token_id
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    start = time.time()
    train(args, model, tokenizer, train_data, valid_data, ignore_idx, args.model_path)
    print('total time: ', (time.time() - start) / 60, " minutes", end='\n\n')

    print('Saving trained model...')
    model_file = os.path.join(args.model_dir,
                              'model_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.bin'
                              .format(len(train_data), args.num_train_epochs))

    config_file = os.path.join(args.model_dir,
                               'config_data{}_trained_after_{}_epochs_only_sum_loss_ignr_pad.json'
                               .format(len(train_data), args.num_train_epochs))
    torch.save(model.state_dict(), model_file)
    model.config.to_json_file(config_file)

    evaluate(args, model, ignore_index=ignore_idx, eval_dataset=valid_data)
