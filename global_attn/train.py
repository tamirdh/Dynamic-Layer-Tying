import datetime

import torch.cuda
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from loaders import preprocess_causal_lm
from models import Transformer
from modified_gpt2_model import IterativeGPTConfig, IterativeGPT
from vanilla_models import VanillaTransformer
from gpt2_model import GPT, GPTConfig
import argparse
from accelerate.utils import LoggerType
from torchinfo import summary


def setup(args, accelerator):
    tokenizer_id = args.token
    accelerator.print(f'Using {tokenizer_id}\'s tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if args.vanilla:
        model = VanillaTransformer(len(tokenizer), args.emsize, args.nhead, args.nlayers, args.emsize * 4,
                                   args.bptt + 1, 0.2, 0.2)
    elif args.gpt:
        config = GPTConfig(args.bptt + 1, len(tokenizer), args.nlayers, args.nhead, args.emsize, 0.2)
        model = GPT(config)
    else:
        # model = Transformer(len(tokenizer), args.emsize, args.nhead, args.emsize * 4, 0.2, accelerator.device, args.k,
        #                     args.nlayers, args.bptt + 1, new=True)

        config = IterativeGPTConfig(args.bptt + 1, len(tokenizer), args.nlayers, args.nhead, args.emsize, 0.2, k=args.k, factor=args.factor)
        model = IterativeGPT(config)
    model = model.bfloat16()
    # model = torch.compile(model)
    return model, tokenizer


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')
    parser.add_argument('--bptt', type=int, default=1024, help='sequence length (used only in wiki2)')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--dataset', type=str, help='Dataset to use',
                        choices=['wiki2', "wiki103", "1b", "lambada", "ptb", "shake"])
    parser.add_argument('--nhead', type=int, default=16, help='number of attention heads')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of decoder blocks')
    parser.add_argument('--emsize', type=int, default=1600, help='embedding dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--token', default='gpt2-xl')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--k', type=int, default=4, help="switch layer")
    parser.add_argument('--factor', type=float, default=0.75)
    parser.add_argument('--exp', help="Experiment name")
    parser.add_argument("--vanilla", action='store_true', help="Use a vanilla version of Transformers")
    parser.add_argument("--gpt", action='store_true', help="Use a GPT version of Transformers")
    args = parser.parse_args()
    return args


def train_language_modeling(model, train_dataset, eval_dataset, tokenizer, args, accelerator: Accelerator):
    lr = args.lr
    device = accelerator.device
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    ignore = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignore)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
    model.to(device)
    if accelerator.is_local_main_process:
        summary(model, input_data=train_dataset[0]['input_ids'].unsqueeze(0).to(accelerator.device), depth=3)
    accelerator.register_for_checkpointing(model)
    accelerator.print(f'LEN TRAIN: {len(train_loader.dataset)}')
    accelerator.print(f'LEN TEST: {len(test_loader.dataset)}')
    best_ppl = float('inf')
    for epoch in range(args.epochs):
        model.train()
        bar = tqdm(train_loader, desc="TRAIN", leave=True, disable=not accelerator.is_local_main_process)
        counter = 1
        train_loss = 0.0
        avg_grad = list()
        max_grad = 0.0
        for batch in bar:
            optimizer.zero_grad()
            inputs = batch['input_ids']
            targets = batch['labels']
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = model(inputs)
            loss = loss_function(outputs.reshape(-1, len(tokenizer)), targets.reshape(-1))
            # Store the current experience in the replay buffer

            train_loss += loss.item()
            accelerator.backward(loss)
            avg_grads = []
            max_grads = []
            for p in model.parameters():
                if p.grad is not None:
                    avg_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())

            _avg_grad = torch.mean(torch.stack(avg_grads))
            _max_grad = max(max_grads)
            if _max_grad > max_grad:
                max_grad = _max_grad
            avg_grad.append(_avg_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            bar.set_postfix(loss=train_loss / counter)
            counter += 1
        accelerator.log({"train_loss": train_loss / counter}, step=epoch + 1)
        accelerator.log({"gradient/mean": torch.mean(torch.stack(avg_grad)).item()}, step=epoch + 1)
        accelerator.log({"gradient/max": max_grad.item()}, step=epoch + 1)
        model.eval()
        with torch.no_grad():
            for ds, loader in [('test', test_loader)]:
                eval_total_loss = 0.0
                eval_acc = 0.0
                counter = 0
                bar = tqdm(loader, desc=f"EVAL-{ds}", leave=True, disable=not accelerator.is_local_main_process)
                for batch in bar:
                    inputs = batch['input_ids']
                    targets = batch['labels']
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs, _ = model(inputs)
                    all_predictions, all_targets = accelerator.gather_for_metrics(
                        (outputs.contiguous(), targets.contiguous()))
                    loss_eval = loss_function(all_predictions.reshape(-1, len(tokenizer)), all_targets.reshape(-1))
                    eval_total_loss += loss_eval.item()
                    preds = torch.argmax(outputs, dim=-1)
                    acc = (preds.reshape(-1) == targets.reshape(-1)).sum()
                    acc = acc.item() / (targets.shape[0] * targets.shape[1])
                    eval_acc += acc
                    counter += 1
                generated_text = gen_text(model, tokenizer, "Most of what is known of Du Fu 's life comes", device)[0]
                accelerator.print(f"\nText:{generated_text}\n")
                perplexity_score = torch.exp(torch.tensor([eval_total_loss / counter])).item()
                if perplexity_score < best_ppl:
                    best_ppl = perplexity_score
                accelerator.log({'Generated_Text': generated_text}, step=epoch + 1)
                accelerator.log({f"perplexity_score ({ds})": perplexity_score}, step=epoch + 1)
                accelerator.log({f"Best PPL ({ds})": best_ppl}, step=epoch + 1)
                accelerator.print(f'Post epoch {epoch + 1} ({ds}): {perplexity_score}')
                accelerator.print(f'Accuracy: {eval_acc / counter}')
            # accelerator.save_state()
            if eval_acc / counter >= 0.99:
                break


def gen_text(model, tokenizer, text, device):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generated = model.module.generate(input_ids, 20)
    generated_text = tokenizer.batch_decode(generated)
    return generated_text


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
    if args.vanilla:
        model_name = "Vanilla"
    elif args.gpt:
        model_name = "GPT"
    else:
        model_name = "Mine"
    now = datetime.datetime.now()
    proj_config = ProjectConfiguration(
        project_dir=f"./exp/{model_name}-{args.exp}-{now}",
        automatic_checkpoint_naming=True,
        total_limit=4)
    accelerator = Accelerator(project_config=proj_config, log_with=LoggerType.TENSORBOARD)
    accelerator.print(f'Got {torch.cuda.device_count()} gpus')
    torch.cuda.empty_cache()
    accelerator.print(f'device={accelerator.device}')
    hps = {"epochs": args.epochs, "learning_rate": args.lr, "bptt": args.bptt}
    accelerator.init_trackers(f'{model_name}-{now}',
                              config=hps)
    model, tokenizer = setup(args, accelerator)
    # count_parameters(model, accelerator)
    accelerator.print(f"Using {len(tokenizer):,} different tokens")
    with accelerator.main_process_first():
        train_ds, test_ds = preprocess_causal_lm(tokenizer, args.bptt, args.dataset)
    train_ds = train_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")
    if args.overfit:
        train_ds = torch.utils.data.Subset(train_ds, list(range(10)))
        test_ds = torch.utils.data.Subset(train_ds, [0])
    train_language_modeling(model, train_ds, test_ds, tokenizer, args, accelerator)
    accelerator.end_training()


if __name__ == '__main__':
    main()
