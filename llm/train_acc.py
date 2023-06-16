import torch.cuda
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from loaders import preprocess_causal_lm
from models import TransformerModel
import argparse


def setup(args, accelerator):
    tokenizer_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    accelerator.print(f'Setting embedder from {tokenizer_id}')
    model = torch.compile(TransformerModel(len(tokenizer), args.emsize, args.nhead, args.d_hid, args.nlayers, 0.1))
    return model, tokenizer


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')
    parser.add_argument('--bptt', type=int, default=512, help='sequence length (used only in wiki2)')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--dataset', type=str, help='Dataset to use',
                        choices=['wiki2', "wiki103", "1b", "lambada", "ptb"])
    parser.add_argument('--nhead', type=int, default=12, help='number of heads in ``nn.MultiheadAttention``')
    parser.add_argument('--d_hid', type=int, default=3072,
                        help='dimension of the feedforward network model in ``nn.TransformerEncoder``')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``')
    parser.add_argument('--emsize', type=int, default=768, help='embedding dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    args = parser.parse_args()
    return args


def train_language_modeling(model, train_dataset, eval_dataset, tokenizer, args, accelerator:Accelerator):
    lr = args.lr
    device = accelerator.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ignore = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=True)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    model.to(device)
    accelerator.register_for_checkpointing(model)
    accelerator.print(f'LEN TRAIN: {len(train_loader.dataset)}')
    accelerator.print(f'LEN TEST: {len(test_loader.dataset)}')
    mask = torch.nn.Transformer.generate_square_subsequent_mask(args.bptt + 1, accelerator.device)
    for epoch in range(args.epochs):
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
            outputs = model(inputs, mask)
            loss = loss_function(outputs.view(-1, len(tokenizer)), targets.view(-1))
            train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()
            bar.set_postfix(loss=train_loss / counter)
            counter += 1
        accelerator.log({"train_loss": train_loss / counter}, step=epoch + 1)
        model.eval()
        eval_total_loss = 0.0
        counter = 0
        for ds, loader in [('test', test_loader)]:
            for batch in tqdm(loader, desc=f"EVAL-{ds}"):
                inputs = batch['input_ids']
                targets = batch['labels']
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs, mask)
                all_predictions, all_targets = accelerator.gather_for_metrics((outputs, targets))
                loss_eval = loss_function(all_predictions.view(-1, len(tokenizer)), all_targets.view(-1))
                eval_total_loss += loss_eval.item()
                counter += 1
            perplexity_score = torch.exp(torch.tensor([eval_total_loss / counter])).item()
            accelerator.log({f"perplexity_score ({ds})": perplexity_score}, step=epoch + 1)
            accelerator.print(f'Post epoch {epoch + 1} ({ds}): {perplexity_score}')
        accelerator.save_state(output_dir=f"./{args.dataset}-check")


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
    accelerator = Accelerator(project_dir=f"./exp/{args.dataset}-exp", log_with="all")
    accelerator.print(f'Got {torch.cuda.device_count()} gpus')
    torch.cuda.empty_cache()
    accelerator.print(f'device={accelerator.device}')
    hps = {"epochs": args.epochs, "learning_rate": args.lr, "bptt": args.bptt}
    accelerator.init_trackers(f'{args.dataset}-track', config=hps)
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
