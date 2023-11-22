import json
from typing import Optional

import transformers

from quantum_models import CustomVit
import torch
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from accelerate.utils import ProjectConfiguration, LoggerType
from accelerate import Accelerator
from loaders import preprocess_images
from torchinfo import summary
from tqdm import tqdm
import torch.distributed as dist
import os



def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--dataset', type=str, help='Dataset to use',
                        choices=['cifar'])
    parser.add_argument('--nhead', type=int, default=16, help='number of attention heads')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of decoder blocks')
    parser.add_argument('--emsize', type=int, default=1600, help='embedding dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--exp', help="Experiment name")
    parser.add_argument('--dqn', action='store_true', help="Use DQN or normal training")
    parser.add_argument("--hidden", type=int, default=5120, help="Hidden layer size")
    args = parser.parse_args()
    return args


def get_dqn_net(input_dim: int, output_dim: int):
    # Input: state vector, depends on the number of layers
    # Output: Action vector, number of layers ** 2
    net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))
    return net


def epsilon_greedy_action_selector(qnet: torch.nn.Module, n_layers: int, layer: int, state: torch.Tensor,
                                   epsilon, device, episode: int, ablation=False) -> list:
    if ablation:
        curr_state = state.tolist()[layer]
        action_vector = curr_state + 1
        if action_vector > layer:
            action_vector = 0
        return action_vector
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


def q_learn_loop(target_model: CustomVit, qnet: Optional[nn.Module], train_ds: Dataset, test_ds: Dataset, args,
                 accelerator: Accelerator, n_classes: int):
    not_dqn = qnet is None
    device = accelerator.device
    # prepare optimizers and dataloaders
    qnet_lr = 0.001
    qnet_gamma = 0.99
    lr = args.lr
    target_opt = torch.optim.AdamW(target_model.parameters(), lr=lr, weight_decay=0.01)
    q_opt = None if not_dqn else torch.optim.AdamW(qnet.parameters(), lr=qnet_lr, weight_decay=0.01)
    loss_function = torch.nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)
    # pass through accelerator for multi-gpu support
    target_opt, train_loader, test_loader, target_model = accelerator.prepare(target_opt,
                                                                              train_loader, test_loader,
                                                                              target_model)
    if not not_dqn:
        qnet, q_opt = accelerator.prepare(qnet, q_opt)
    target_model.to(device)
    is_distributed = isinstance(target_model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    if accelerator.is_local_main_process:
        summary(target_model, input_data=train_ds[0]['pixel_values'].unsqueeze(0).to(accelerator.device), depth=3)
    # type hinting for easier completions
    target_model: CustomVit
    epsilon = 1.0
    state = target_model.module.get_state() if is_distributed else target_model.get_state()
    best_acc = 0.0

    for episode in range(1, args.epochs + 1):
        for layer in range(1, args.nlayers):
            if accelerator.is_local_main_process and not not_dqn:
                action = epsilon_greedy_action_selector(qnet, args.nlayers, layer, state, epsilon, device, episode,
                                                        ablation=False)
                action_tensor = torch.tensor([action], dtype=torch.int64).to(device)  # Convert int to tensor
                state_dict = {int(key): int(value) for key, value in enumerate(state)}
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
            if is_distributed and not not_dqn:
                target_model.module.apply_action(action, layer)
            else:
                if not not_dqn:
                    target_model.apply_action(action, layer)
            # target_model.module.enforce_weights() if is_distributed else target_model.enforce_weights()
            target_model.train()
            if not not_dqn:
                target_model = accelerator.prepare(target_model.module) if is_distributed else accelerator.prepare(
                    target_model)
            target_model.to(device)
            # multiple training steps per episode
            episode_loss = 0.0
            limit = 10
            counter = 0
            bar = tqdm(train_loader, desc=f"train layer {layer}", leave=True,
                       disable=not accelerator.is_local_main_process,
                       total=limit if not not_dqn else None)
            for i, batch in enumerate(bar):
                if i >= limit and not not_dqn:
                    break
                target_opt.zero_grad()
                inputs = batch['pixel_values']
                targets = batch['labels']
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs, _ = target_model(inputs)
                loss = loss_function(outputs.reshape(-1, n_classes), targets.reshape(-1))
                episode_loss += loss.item()
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
                target_opt.step()
                counter += 1

            accelerator.log({f"train_loss": episode_loss / counter}, step=episode * args.nlayers + layer)
            formatted_state = ", ".join([f"{int(key)} <- {int(value)}" for key, value in enumerate(state)])
            accelerator.print(json.dumps({"train_loss": episode_loss / counter, "episode": episode, "layer": layer,
                                          "state": formatted_state, "action": action}, indent=4))
            target_model.eval()
            # evaluate Q-function
            if not not_dqn:
                q_opt.zero_grad()

                for iterator in train_loader:
                    batch = iterator
                    break
                inputs = batch['pixel_values']
                targets = batch['labels']
                reward = target_model.module.evaluate_on_samples(inputs, targets, loss_function, 0) \
                    if is_distributed else target_model.evaluate_on_samples(inputs, targets, loss_function, 0)
                next_state = target_model.module.get_state() if is_distributed else target_model.get_state()
                # Compute the target Q-value
                next_state = next_state.to(device)
                with torch.no_grad():
                    start_index = 0
                    end_index = 1
                    for i in range(1, layer + 1):
                        start_index += i
                        end_index += i + 1
                    target = reward + qnet_gamma * torch.max(qnet(next_state)[start_index:end_index])
                # Compute the predicted Q-value
                predicted = qnet(state.to(device))[start_index:end_index][action]
                # Update Q-network
                q_loss = criterion(predicted.to(device), target.to(device))
                accelerator.log({f"DQN loss": q_loss.item()}, step=episode * args.nlayers + layer)
                accelerator.backward(q_loss)
                q_opt.step()
                state = next_state
            if not_dqn:
                break
        n_trainable = sum(p.numel() for p in target_model.parameters() if p.requires_grad) / 1000000
        accelerator.log({"# trainable params (M)": n_trainable}, step=episode)
        accelerator.log({"epsilon": epsilon}, step=episode)
        epsilon = max(epsilon * 0.95, 0.1)

        # evaluate PPL and accuracy
        with torch.no_grad():
            for ds, loader in [('test', test_loader)]:
                eval_total_loss = 0.0
                eval_acc = 0.0
                counter = 0
                for batch in loader:
                    inputs = batch['pixel_values']
                    targets = batch['labels']
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs, _ = target_model(inputs)
                    all_predictions, all_targets = accelerator.gather_for_metrics(
                        (outputs.contiguous(), targets.contiguous()))
                    loss_eval = loss_function(all_predictions.reshape(-1, n_classes), all_targets.reshape(-1))
                    eval_total_loss += loss_eval.item()
                    preds = torch.argmax(outputs, dim=-1)
                    acc = (preds.reshape(-1) == targets.reshape(-1)).sum()
                    acc = acc.item() / (targets.shape[0])
                    eval_acc += acc
                    counter += 1
                accuracy_score = torch.tensor([eval_acc / counter]).item()
                if accuracy_score > best_acc:
                    best_acc = accuracy_score

                accelerator.log({f"accuracy score ({ds})": accuracy_score}, step=episode + 1)
                accelerator.log({f"Best PPL ({ds})": best_acc}, step=episode + 1)
                accelerator.print(f'Post epoch {episode + 1} ({ds}): {accuracy_score}')
                accelerator.print(f'Accuracy: {eval_acc / counter}')


def setup_training(args: argparse.Namespace):
    # set up training environment
    is_dqn = args.dqn
    model_name = "DQN" if is_dqn else "VIT"
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
    hps = {"epochs": args.epochs, "learning_rate": args.lr}
    accelerator.init_trackers(f'{model_name}-{now}',
                              config=hps)
    # set up target model & dataset

    config = transformers.ViTConfig(args.emsize, args.nlayers, args.nhead, intermediate_size=args.hidden, num_labels=10)
    # model = DynamicGPT(config) if not args.llama else CustomBert(BertConfig(len(tokenizer), is_decoder=True))
    model = CustomVit(config)
    if is_dqn:
        model.custom_post_init()
    model = model.bfloat16() if 'cuda' in accelerator.device.type else model
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = transformers.ViTImageProcessor.from_pretrained(model_name_or_path)
    with accelerator.main_process_first():
        train_ds, test_ds = preprocess_images(args.dataset, processor)
    train_ds = train_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")
    n_classes = 10
    accelerator.print(f"Using {n_classes:,} different classes")
    if is_dqn:
        q_net = get_dqn_net(args.nlayers, sum(range(1, args.nlayers + 1)))
        q_learn_loop(model, q_net, train_ds, test_ds, args, accelerator, n_classes)
    else:
        q_learn_loop(model, None, train_ds, test_ds, args, accelerator, n_classes)


if __name__ == '__main__':
    arguments = init_args()
    setup_training(arguments)
