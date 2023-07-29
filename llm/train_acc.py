import torch.cuda
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from loaders import preprocess_causal_lm
from models import SelfFeedConfig, SelfFeedLMHeadModel, MyModel
import argparse
from accelerate.utils import LoggerType


def setup(args, accelerator):
    tokenizer_id = "gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    accelerator.print(f'Setting embedder from {tokenizer_id}')
    config = transformers.AutoConfig.from_pretrained(tokenizer_id)
    config.niters=args.niters
    config.use_pretrained_model = tokenizer_id
    config.max_layers = args.nlayers
    # model = SelfFeedLMHeadModel(config)
    model = MyModel(config, freeze_base=False)
    # model = transformers.AutoModelForCausalLM.from_pretrained("gpt2-xl")
    return model, tokenizer


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')
    parser.add_argument('--bptt', type=int, default=1024, help='sequence length (used only in wiki2)')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--dataset', type=str, help='Dataset to use',
                        choices=['wiki2', "wiki103", "1b", "lambada", "ptb"])
    parser.add_argument('--nhead', type=int, default=16, help='number of attention heads')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of decoder blocks')
    parser.add_argument('--niters', type=int, default=4,
                        help='number of inner iterations inside each decoder block')
    parser.add_argument('--emsize', type=int, default=1600, help='embedding dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    args = parser.parse_args()
    return args


def train_language_modeling(model, train_dataset, eval_dataset, tokenizer, args, accelerator: Accelerator):
    lr = args.lr
    device = accelerator.device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    ignore = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    model.to(device)
    accelerator.register_for_checkpointing(model)
    accelerator.print(f'LEN TRAIN: {len(train_loader.dataset)}')
    accelerator.print(f'LEN TEST: {len(test_loader.dataset)}')
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        model.train()
        bar = tqdm(train_loader, desc="TRAIN", leave=True)
        counter = 1
        train_loss = 0.0

        for batch in bar:
            optimizer.zero_grad()
            inputs = batch['input_ids']
            targets = batch['labels']
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs, attention_mask=batch['attention_mask'].to(device)).logits
            loss = loss_function(outputs.view(-1, len(tokenizer)), targets.view(-1))
            train_loss += loss.item()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bar.set_postfix(loss=train_loss / counter)
            counter += 1
        accelerator.log({"train_loss": train_loss / counter}, step=epoch + 1)
        model.eval()
        with torch.no_grad():
            for ds, loader in [('test', test_loader)]:
                torch.cuda.empty_cache()
                eval_total_loss = 0.0
                counter = 0
                for batch in tqdm(loader, desc=f"EVAL-{ds}"):
                    inputs = batch['input_ids']
                    targets = batch['labels']
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs, attention_mask=batch['attention_mask'].to(device)).logits
                    all_predictions, all_targets = accelerator.gather_for_metrics((outputs, targets))
                    loss_eval = loss_function(all_predictions.view(-1, len(tokenizer)), all_targets.view(-1))
                    eval_total_loss += loss_eval.item()
                    counter += 1
                perplexity_score = torch.exp(torch.tensor([eval_total_loss / counter])).item()
                accelerator.log({f"perplexity_score ({ds})": perplexity_score}, step=epoch + 1)
                accelerator.print(f'Post epoch {epoch + 1} ({ds}): {perplexity_score}')
            accelerator.save_state(output_dir=f"./{args.dataset}-{args.nlayers}-{args.niters}-new-check")


def count_parameters(model, accelerator: Accelerator):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            accelerator.print(f"{name}: {num_params:,}")

    accelerator.print(f"Total trainable parameters: {total_params:,}")


def main():
    args = init_args()
    accelerator = Accelerator(project_dir=f"./exp/{args.dataset}-{args.nlayers}-{args.niters}-new-exp",
                              log_with=LoggerType.TENSORBOARD)
    accelerator.print(f'Got {torch.cuda.device_count()} gpus')
    torch.cuda.empty_cache()
    accelerator.print(f'device={accelerator.device}')
    hps = {"epochs": args.epochs, "learning_rate": args.lr, "bptt": args.bptt}
    accelerator.init_trackers(f'{args.dataset}-{args.nlayers}-new-track', config=hps)
    model, tokenizer = setup(args, accelerator)
    count_parameters(model, accelerator)
    accelerator.print(f"Using {len(tokenizer):,} different tokens")
    with accelerator.main_process_first():
        train_ds, test_ds = preprocess_causal_lm(tokenizer, args.bptt, args.dataset)
    train_ds = train_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")
    train_language_modeling(model, train_ds, test_ds, tokenizer, args, accelerator)
    accelerator.end_training()


if __name__ == '__main__':
    main()
