import json

from quantum_models import DynamicGPT, GPTConfig, CustomLLAMA2
import torch
import datetime
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from accelerate.utils import ProjectConfiguration, LoggerType
from accelerate import Accelerator
from loaders import preprocess_causal_lm
from torchinfo import summary
from tqdm import tqdm
import torch.distributed as dist


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
    parser.add_argument('--exp', help="Experiment name")
    parser.add_argument('--llama', action='store_true', help="Use llama model")
    args = parser.parse_args()
    return args


def get_dqn_net(input_dim: int, output_dim: int):
    # Input: state vector, depends on the number of layers
    # Output: Action vector, number of layers ** 2
    net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))
    return net


def epsilon_greedy_action_selector(qnet: torch.nn.Module, n_layers: int, layer: int, state: torch.Tensor,
                                   epsilon, device) -> list:
    if np.random.rand() < epsilon:
        action_vector = np.random.randint(-1, high=layer)
    else:
        with torch.no_grad():
            q_values = qnet(state.to(device))
            start_index = 0
            end_index = 1
            for i in range(1, layer + 1):
                start_index += i
                end_index += i + 1
            layer_actions = q_values[start_index: end_index]
            action_vector = torch.argmax(layer_actions).item()
    return action_vector


def q_learn_loop(target_model: DynamicGPT, qnet: nn.Module, train_ds: Dataset, test_ds: Dataset, args,
                 accelerator: Accelerator, tokenizer):
    device = accelerator.device
    # prepare optimizers and dataloaders
    qnet_lr = 0.001
    qnet_gamma = 0.99
    lr = args.lr
    target_opt = torch.optim.AdamW(target_model.parameters(), lr=lr, weight_decay=0.01)
    q_opt = torch.optim.AdamW(qnet.parameters(), lr=qnet_lr, weight_decay=0.01)
    loss_function = torch.nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    # pass through accelerator for multi-gpu support
    target_opt, q_opt, train_loader, test_loader, target_model, qnet = accelerator.prepare(target_opt, q_opt,
                                                                                           train_loader, test_loader,
                                                                                           target_model, qnet)
    is_distributed = isinstance(target_model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    if accelerator.is_local_main_process:
        summary(target_model, input_data=train_ds[0]['input_ids'].unsqueeze(0).to(accelerator.device), depth=3)
    # type hinting for easier completions
    target_model: DynamicGPT
    epsilon = 1.0
    state = target_model.module.get_state() if is_distributed else target_model.get_state()
    best_ppl = float('inf')
    for episode in range(1, args.epochs + 1):
        for layer in range(1, args.nlayers):
            if accelerator.is_local_main_process:
                action = epsilon_greedy_action_selector(qnet, args.nlayers, layer, state, epsilon, device)
                action_tensor = torch.tensor([action], dtype=torch.int64).to(device)  # Convert int to tensor
                state_dict = {int(key): int(value) for key,value in enumerate(state)}
                # if the state key == value then the layer is trainable
                number_of_trainable_layers = sum([1 for key, value in state_dict.items() if key == value])
                with open(f"states-{args.exp}.txt", "a") as f:
                    f.write(json.dumps(state_dict) + "\n")
                with open(f"actions-{args.exp}.txt", "a") as f:
                    f.write(f"{action_tensor.item()}\n")
                with open(f"trainable-{args.exp}.txt", "a") as f:
                    f.write(f"{number_of_trainable_layers}\n")
            else:
                action_tensor = torch.zeros([1], dtype=torch.int64).to(device)
            if is_distributed:
                dist.broadcast(action_tensor, src=0)
            action = action_tensor.item()
            if is_distributed:
                target_model.module.apply_action(action, layer)
            else:
                target_model.apply_action(action, layer)
            target_model.train()

            target_model = accelerator.prepare(target_model.module) if is_distributed else accelerator.prepare(
                target_model)
            # multiple training steps per episode
            episode_loss = 0.0
            limit = 10
            bar = tqdm(train_loader, desc=f"train layer {layer}", leave=True,
                       disable=not accelerator.is_local_main_process,
                       total=limit)
            for i, batch in enumerate(bar):
                if i >= limit:
                    break
                target_opt.zero_grad()
                inputs = batch['input_ids']
                targets = batch['labels']
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, _ = target_model(inputs)
                loss = loss_function(outputs.reshape(-1, len(tokenizer)), targets.reshape(-1))
                episode_loss += loss.item()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
                target_opt.step()

            accelerator.log({f"train_loss": episode_loss / limit}, step=episode * args.nlayers + layer)
            formatted_state = ", ".join([f"{int(key)} <- {int(value)}" for key, value in enumerate(state)])
            accelerator.print(json.dumps({"train_loss": episode_loss / limit, "episode": episode, "layer": layer,
                                          "state": formatted_state, "action": action}, indent=4))
            target_model.eval()
            # evaluate Q-function
            q_opt.zero_grad()

            for iterator in train_loader:
                batch = iterator
                break
            inputs = batch['input_ids']
            targets = batch['labels']
            reward = target_model.module.evaluate_on_samples(inputs, targets, loss_function, len(tokenizer)) \
                if is_distributed else target_model.evaluate_on_samples(inputs, targets, loss_function, len(tokenizer))
            next_state = target_model.module.get_state() if is_distributed else target_model.get_state()
            # Compute the target Q-value
            next_state = next_state.to(device)
            with torch.no_grad():
                start_index = 0
                end_index = 1
                for i in range(1, layer + 1):
                    start_index += i
                    end_index += i + 1
                target = reward.clip(-10, 10) + qnet_gamma * torch.max(qnet(next_state)[start_index:end_index])
            # Compute the predicted Q-value
            predicted = qnet(state.to(device))[start_index:end_index][action]
            # Update Q-network
            q_loss = criterion(predicted, target)
            accelerator.log({f"DQN loss": q_loss.item()}, step=episode * args.nlayers + layer)
            accelerator.backward(q_loss)
            q_opt.step()
            state = next_state
        n_trainable = sum(p.numel() for p in target_model.parameters() if p.requires_grad) / 1000000
        accelerator.log({"# trainable params (M)": n_trainable}, step=episode)
        epsilon = max(epsilon * 0.95, 0.01)

        # evaluate PPL and accuracy
        with torch.no_grad():
            for ds, loader in [('test', test_loader)]:
                eval_total_loss = 0.0
                eval_acc = 0.0
                counter = 0
                for batch in loader:
                    inputs = batch['input_ids']
                    targets = batch['labels']
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs, _ = target_model(inputs)
                    all_predictions, all_targets = accelerator.gather_for_metrics(
                        (outputs.contiguous(), targets.contiguous()))
                    loss_eval = loss_function(all_predictions.reshape(-1, len(tokenizer)), all_targets.reshape(-1))
                    eval_total_loss += loss_eval.item()
                    preds = torch.argmax(outputs, dim=-1)
                    acc = (preds.reshape(-1) == targets.reshape(-1)).sum()
                    acc = acc.item() / (targets.shape[0] * targets.shape[1])
                    eval_acc += acc
                    counter += 1
                perplexity_score = torch.exp(torch.tensor([eval_total_loss / counter])).item()
                if perplexity_score < best_ppl:
                    best_ppl = perplexity_score
                accelerator.log({f"perplexity_score ({ds})": perplexity_score}, step=episode + 1)
                accelerator.log({f"Best PPL ({ds})": best_ppl}, step=episode + 1)
                accelerator.print(f'Post epoch {episode + 1} ({ds}): {perplexity_score}')
                accelerator.print(f'Accuracy: {eval_acc / counter}')


def unravel_index(index, nlayers):
    action_vector = []
    for i in range(nlayers, 0, -1):
        choice = index % i
        action_vector.insert(0, choice)
        index //= i
    return action_vector


def ravel_index(action_vector, nlayers):
    index = 0
    offset = 0
    for i in range(1, nlayers + 1):
        index += action_vector[i - 1] + offset
        offset += i
    return index


def setup_training(args: argparse.Namespace):
    # set up training environment
    model_name = "Qlearning"
    now = datetime.datetime.now()
    proj_config = ProjectConfiguration(
        project_dir=f"./exp/{model_name}-{args.exp}-{now}",
        automatic_checkpoint_naming=True,
        total_limit=4)
    accelerator = Accelerator(project_config=proj_config, log_with=LoggerType.TENSORBOARD,
                              cpu=not torch.cuda.is_available())
    accelerator.print(f'Got {torch.cuda.device_count()} gpus')
    torch.cuda.empty_cache()
    accelerator.print(f'device={accelerator.device}')
    hps = {"epochs": args.epochs, "learning_rate": args.lr, "bptt": args.bptt}
    accelerator.init_trackers(f'{model_name}-{now}',
                              config=hps)
    # set up target model & dataset
    tokenizer_id = args.token if not args.llama else 'meta-llama/Llama-2-7b-hf'
    accelerator.print(f'Using {tokenizer_id}\'s tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    config = GPTConfig(args.bptt + 1, len(tokenizer), args.nlayers, args.nhead, args.emsize, 0.2)
    model = DynamicGPT(config) if not args.llama else CustomLLAMA2.from_pretrained(tokenizer_id)
    # model = model.bfloat16() if 'cuda' in accelerator.device.type else model
    accelerator.print(f"Using {len(tokenizer):,} different tokens")
    with accelerator.main_process_first():
        train_ds, test_ds = preprocess_causal_lm(tokenizer, args.bptt, args.dataset)
    train_ds = train_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")
    # set up Q-network
    if args.llama:
        args.nlayers = len(model.model.layers)
        print(f"Using {args.nlayers} layers")
    q_net = get_dqn_net(args.nlayers, sum(range(1, args.nlayers + 1)))
    q_learn_loop(model, q_net, train_ds, test_ds, args, accelerator, tokenizer)


if __name__ == '__main__':
    arguments = init_args()
    setup_training(arguments)
