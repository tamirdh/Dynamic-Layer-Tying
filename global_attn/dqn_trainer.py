import json

import transformers

from quantum_models import DynamicGPT, GPTConfig, CustomBert, CustomLLAMA2
import torch
import datetime
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
from accelerate.utils import ProjectConfiguration, LoggerType
from accelerate import Accelerator
from loaders import preprocess_causal_lm, preprocess_squad, preprocess_glue_sst2, preprocess_glue_single_and_pair
from torchinfo import summary
from tqdm import tqdm
import torch.distributed as dist
from transformers import BertConfig, T5ForConditionalGeneration, T5Config
import os


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, help='evaluation batch size')
    parser.add_argument('--bptt', type=int, default=1024, help='sequence length (used only in wiki2)')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--dataset', type=str, help='Dataset to use',
                        choices=['wiki2', "wiki103", "1b", "lambada", "ptb", "shake", "squad", "sst2", "cola", "qnli",
                                 "mnli", "mrpc", "qqp", "rte", "wnli"])
    parser.add_argument('--nhead', type=int, default=16, help='number of attention heads')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of decoder blocks')
    parser.add_argument('--emsize', type=int, default=1600, help='embedding dimension')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--token', default='gpt2-xl')
    parser.add_argument('--exp', help="Experiment name")
    parser.add_argument('--llama', action='store_true', help="Use llama model")
    parser.add_argument('--vnl', action='store_true', help="Use vanilla model (all layers are trainable)")
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
        # curr_state = state.tolist()[layer]
        # action_vector = curr_state + 1
        # if action_vector > layer:
        #     action_vector = 0
        # return action_vector
        # Sequence ablation
        # if layer % 2 == 0:
        #     action_vector = layer
        # else:
        #     action_vector = layer - 1
        # Rev ablation
        if layer < n_layers // 2:
            action_vector = layer
        else:
            action_vector = n_layers - layer - 1
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
    target_model.to(device)
    is_distributed = isinstance(target_model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    if accelerator.is_local_main_process:
        summary(target_model, input_data=train_ds[0]['input_ids'].unsqueeze(0).to(accelerator.device), depth=3)
    # type hinting for easier completions
    target_model: DynamicGPT
    epsilon = 1.0
    state = target_model.module.get_state() if is_distributed else target_model.get_state()
    best_ppl = float('inf')

    for episode in range(1, args.epochs + 1):
        total_memory_used = 0

        # Iterate over all GPUs and sum their memory allocations
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            torch.cuda.set_device(gpu_id)
            total_memory_used += torch.cuda.memory_allocated()

        # Convert the total memory from bytes to megabytes
        total_memory_used_mb = total_memory_used / 1000000

        # Write the total memory used to the file
        with open(f'mem-{args.exp}.txt', 'a') as f:
            f.write(f"{total_memory_used_mb}\n")
        for layer in range(1, args.nlayers):
            if accelerator.is_local_main_process:
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
            if is_distributed:
                target_model.module.apply_action(action, layer)
            else:
                target_model.apply_action(action, layer)
            # target_model.module.enforce_weights() if is_distributed else target_model.enforce_weights()
            target_model.train()

            target_model = accelerator.prepare(target_model.module) if is_distributed else accelerator.prepare(
                target_model)
            target_model.to(device)
            # multiple training steps per episode
            episode_loss = 0.0
            limit = 4
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
            q_loss = criterion(predicted.to(device), target.to(device))
            accelerator.log({f"DQN loss": q_loss.item()}, step=episode * args.nlayers + layer)
            accelerator.backward(q_loss)
            q_opt.step()
            state = next_state
        n_trainable = sum(p.numel() for p in target_model.parameters() if p.requires_grad) / 1000000
        accelerator.log({"# trainable params (M)": n_trainable}, step=episode)
        accelerator.log({"epsilon": epsilon}, step=episode)
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
                if accelerator.is_local_main_process and perplexity_score <= best_ppl:
                    accelerator.print("Saving model at ")
                    accelerator.save(target_model, f"./models/{args.exp}-best_ppl-{ds}.pt")
                accelerator.log({f"perplexity_score ({ds})": perplexity_score}, step=episode + 1)
                accelerator.log({f"Best PPL ({ds})": best_ppl}, step=episode + 1)
                accelerator.print(f'Post epoch {episode + 1} ({ds}): {perplexity_score}')
                accelerator.print(f'Accuracy: {eval_acc / counter}')


def q_learn_loop_sst2(target_model: DynamicGPT, qnet: nn.Module, train_ds: Dataset, test_ds: Dataset, args,
                      accelerator: Accelerator, tokenizer):
    # [Initialization code remains the same as in q_learn_loop]
    device = accelerator.device
    # prepare optimizers and dataloaders
    qnet_lr = 0.001
    qnet_gamma = 0.99
    lr = args.lr
    target_opt = torch.optim.AdamW(target_model.parameters(), lr=lr, weight_decay=0.01)
    q_opt = torch.optim.AdamW(qnet.parameters(), lr=qnet_lr, weight_decay=0.01)
    loss_function = torch.nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    # pass through accelerator for multi-gpu support
    target_opt, train_loader, test_loader, target_model = accelerator.prepare(target_opt, train_loader,
                                                                              test_loader, target_model)
    target_model.to(device)
    if accelerator.is_local_main_process:
        summary(target_model, input_data=train_ds[0]['input_ids'].unsqueeze(0).to(accelerator.device), depth=3)
    # type hinting for easier completions
    target_model: DynamicGPT
    best_acc = 0.0
    saved_model_path = "models/unwrappedModel.pt"  # Path to the saved model

    for episode in range(1, args.epochs + 1):
        target_model.train()
        episode_loss = 0.0
        bar = tqdm(train_loader, desc=f"Episode {episode}", leave=True,
                   disable=not accelerator.is_local_main_process,
                   total=len(train_loader))
        for i, batch in enumerate(bar):
            target_opt.zero_grad()
            inputs = batch['input_ids']
            targets = batch['labels']
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = target_model(inputs)
            loss = loss_function(outputs[:, -1, :].squeeze(-1), targets.to(dtype=outputs.dtype))
            episode_loss += loss.item()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)
            target_opt.step()
        limit = i + 1
        accelerator.log({f"train_loss": episode_loss / (i + 1)}, step=episode)
        accelerator.print(json.dumps({"train_loss": episode_loss / limit, "episode": episode}, indent=4))
        target_model.eval()
        # evaluate accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in test_loader:
                inputs = batch['input_ids']
                labels = batch['labels']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, _ = target_model(inputs)
                all_predictions, all_targets = accelerator.gather_for_metrics(
                    (outputs.contiguous(), labels.contiguous()))
                predictions = torch.sigmoid(all_predictions[:, -1, :].squeeze(-1))
                predictions = (predictions > 0.5).int()
                correct += (predictions == all_targets).sum().item()
                total += all_targets.size(0)

            accuracy = correct / total
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(target_model.state_dict(), saved_model_path)

            accelerator.log({f"Accuracy": accuracy, "Best Accuracy": best_acc}, step=episode)


def q_learn_loop_multiclass(target_model: DynamicGPT, qnet: nn.Module, train_ds: Dataset, test_ds: Dataset, args,
                            accelerator: Accelerator, tokenizer):
    # [Initialization code remains the same as in q_learn_loop]
    device = accelerator.device
    # prepare optimizers and dataloaders
    lr = args.lr
    target_opt = torch.optim.AdamW(target_model.parameters(), lr=lr, weight_decay=0.01)
    loss_function = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    # pass through accelerator for multi-gpu support
    target_opt, train_loader, test_loader, target_model = accelerator.prepare(target_opt, train_loader,
                                                                              test_loader, target_model)
    target_model.to(device)
    if accelerator.is_local_main_process:
        summary(target_model, input_data=train_ds[0]['input_ids'].unsqueeze(0).to(accelerator.device), depth=3)
    # type hinting for easier completions
    target_model: DynamicGPT
    best_acc = 0.0
    saved_model_path = "models/unwrappedModel.pt"  # Path to the saved model

    for episode in range(1, args.epochs + 1):
        target_model.train()
        episode_loss = 0.0
        bar = tqdm(train_loader, desc=f"Episode {episode}", leave=True,
                   disable=not accelerator.is_local_main_process,
                   total=len(train_loader))
        for i, batch in enumerate(bar):
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
        limit = i + 1
        accelerator.log({f"train_loss": episode_loss / (i + 1)}, step=episode)
        accelerator.print(json.dumps({"train_loss": episode_loss / limit, "episode": episode}, indent=4))
        target_model.eval()
        # evaluate accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in test_loader:
                inputs = batch['input_ids']
                labels = batch['labels']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs, _ = target_model(inputs)
                all_predictions, all_targets = accelerator.gather_for_metrics(
                    (outputs.contiguous(), labels.contiguous()))
                predictions = torch.argmax(all_predictions[:, -1, :], dim=-1)
                correct += (predictions == all_targets).sum().item()
                total += all_targets.size(0)

            accuracy = correct / total
            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(target_model.state_dict(), saved_model_path)

            accelerator.log({f"Accuracy": accuracy, "Best Accuracy": best_acc}, step=episode)


def qlearn_squad_task(target_model: DynamicGPT, qnet: nn.Module, train_ds: Dataset, test_ds: Dataset, args,
                      accelerator: Accelerator, tokenizer):
    """Same as causalLM, but with the SQuAD task"""
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
    target_model.to(device)
    is_distributed = isinstance(target_model, (nn.DataParallel, nn.parallel.DistributedDataParallel))
    if accelerator.is_local_main_process:
        summary(target_model, input_data=train_ds[0]['input_ids'].unsqueeze(0).to(accelerator.device), depth=3)
    # type hinting for easier completions
    target_model: DynamicGPT
    epsilon = 1.0
    state = target_model.module.get_state() if is_distributed else target_model.get_state()
    best_em = 0.0
    best_f1 = 0.0
    for episode in range(1, args.epochs + 1):
        for layer in range(1, args.nlayers):
            if args.vnl and layer > 1:
                break
            if accelerator.is_local_main_process:
                action = epsilon_greedy_action_selector(qnet, args.nlayers, layer, state, epsilon, device, episode,
                                                        ablation=False)
                action_tensor = torch.tensor([action], dtype=torch.int64).to(device)  # Convert int to tensor
                state_dict = {int(key): int(value) for key, value in enumerate(state)}
                # if the state key == value then the layer is trainable
                number_of_trainable_layers = sum([1 for key, value in state_dict.items() if key == value])
            else:
                action_tensor = torch.zeros([1], dtype=torch.int64).to(device)
            if is_distributed:
                dist.broadcast(action_tensor, src=0)
            action = action_tensor.item()
            if args.vnl:
                action = -1
            if is_distributed:
                target_model.module.apply_action(action, layer)
            else:
                target_model.apply_action(action, layer)
            target_model.train()

            target_model = accelerator.prepare(target_model.module) if is_distributed else accelerator.prepare(
                target_model)
            target_model.to(device)
            # multiple training steps per episode
            episode_loss = 0.0
            limit = 15
            bar = tqdm(train_loader, desc=f"train layer {layer}", leave=True,
                       disable=not accelerator.is_local_main_process,
                       total=limit if not args.vnl else len(train_loader))
            for i, batch in enumerate(bar):
                if i >= limit and not args.vnl:
                    break
                target_opt.zero_grad()
                inputs = batch['input_ids']
                start_positions = batch['start_positions']  # Start positions of answers
                end_positions = batch['end_positions']
                inputs = inputs.to(device)
                start_positions = start_positions.to(device)
                end_positions = end_positions.to(device)
                outputs, _ = target_model(inputs)
                start_logits, end_logits = outputs.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)
                start_loss = loss_function(start_logits, start_positions)
                end_loss = loss_function(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                episode_loss += total_loss.item()
                accelerator.backward(total_loss)
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
            targets = (batch['start_positions'], batch['end_positions'])
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
            q_loss = criterion(predicted.to(device), target.to(device))
            accelerator.log({f"DQN loss": q_loss.item()}, step=episode * args.nlayers + layer)
            accelerator.backward(q_loss)
            q_opt.step()
            state = next_state
        n_trainable = sum(p.numel() for p in target_model.parameters() if p.requires_grad) / 1000000
        accelerator.log({"# trainable params (M)": n_trainable}, step=episode)
        accelerator.log({"epsilon": epsilon}, step=episode)
        epsilon = max(epsilon * 0.95, 0.01)

        # evaluate F1 and EM
        with torch.no_grad():
            for ds, loader in [('test', test_loader)]:
                best_em = 0.0
                best_f1 = 0.0
                total_f1 = 0.0
                total_em = 0.0
                counter = 0
                for batch in loader:
                    inputs = batch['input_ids']
                    start_positions = batch['start_positions']  # Start positions of answers
                    end_positions = batch['end_positions']

                    # Move inputs and targets to the device (e.g., GPU)
                    inputs = inputs.to(device)
                    start_positions = start_positions.to(device)
                    end_positions = end_positions.to(device)

                    # Forward pass
                    outputs, _ = target_model(inputs)
                    start_logits, end_logits = outputs.split(1, dim=-1)
                    start_logits = start_logits.squeeze(-1)
                    end_logits = end_logits.squeeze(-1)

                    # Predictions
                    start_preds = torch.argmax(start_logits, dim=1)
                    end_preds = torch.argmax(end_logits, dim=1)

                    exact_match = ((start_preds == start_positions) & (end_preds == end_positions)).all()
                    em_batch = exact_match.float().mean().item()
                    total_em += em_batch

                    # F1 Score for each example in the batch
                    for i in range(start_positions.size(0)):
                        start_pred = start_preds[i]
                        end_pred = end_preds[i]
                        start_true = start_positions[i]
                        end_true = end_positions[i]

                        # True Positives, False Positives, False Negatives
                        tp = (start_pred == start_true) and (end_pred == end_true)
                        fp = not tp
                        fn = (start_pred != start_true) or (end_pred != end_true)

                        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
                        total_f1 += f1
                    counter += 1

                # Averaging the metrics and loss
                avg_f1 = total_f1 / counter
                avg_em = total_em / counter

                # Update best scores
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                if avg_em > best_em:
                    best_em = avg_em

                accelerator.log({f"F1 ({ds})": avg_f1, f"EM ({ds})": avg_em}, step=episode + 1)
                accelerator.log({f"Best F1 ({ds})": best_f1, f"Best EM ({ds}": best_em}, step=episode + 1)
                accelerator.print(f'Post epoch {episode + 1} ({ds}): f1- {avg_f1:.5f}, em- {avg_em:.5f}')


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


def glue_keys_pairs():
    data = {"mrpc": ("sentence1", "sentence2", 2),
            "qqp": ("question1", "question2", 2),
            "rte": ("sentence1", "sentence2", 2),
            "wnli": ("sentence1", "sentence2", 2)}
    return data


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
    tokenizer_id = args.token if not args.llama else 'gpt2-xl'
    accelerator.print(f'Using {tokenizer_id}\'s tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    config = GPTConfig(args.bptt + 1, len(tokenizer), args.nlayers, args.nhead, args.emsize, 0.2)
    # model = DynamicGPT(config) if not args.llama else CustomBert(BertConfig(len(tokenizer), is_decoder=True))
    model = DynamicGPT(config) if not args.llama else CustomLLAMA2(transformers.LlamaConfig(len(tokenizer)))
    if args.llama:
        model.custom_post_init()
    key1, key2, n_classes = None, None, None
    accelerator.print(f"Using {len(tokenizer):,} different tokens")
    with accelerator.main_process_first():
        if args.dataset == "squad":
            model: DynamicGPT = torch.load("models/unwrappedModel.pt", map_location=torch.device('cpu'))
            model.switch_to_squad_head()
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.cls_token = tokenizer.eos_token
            train_ds, test_ds = preprocess_squad(tokenizer, args.dataset, args.bptt)

        elif args.dataset in ["sst2", "cola", "qnli"]:
            loaded = torch.load("models/unwrappedModel.pt", map_location=torch.device('cpu'))
            model: DynamicGPT = loaded
            model.switch_to_sst2_head()
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.cls_token = tokenizer.eos_token
            train_ds, test_ds = preprocess_glue_sst2(tokenizer, args.bptt, args.dataset)
        elif args.dataset in ["mnli", "mrpc", "qqp", "rte", "wnli"]:
            key1, key2, n_classes = glue_keys_pairs()[args.dataset]
            loaded = torch.load("models/unwrappedModel.pt", map_location=torch.device('cpu'))
            model: DynamicGPT = loaded
            if n_classes == 2:
                model.switch_to_sst2_head()
            else:
                model.switch_to_labels_head(n_classes)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.cls_token = tokenizer.eos_token
            train_ds, test_ds = preprocess_glue_single_and_pair(tokenizer, args.bptt, args.dataset, True,
                                                                (key1, key2))


        else:
            train_ds, test_ds = preprocess_causal_lm(tokenizer, args.bptt, args.dataset)
        if args.vnl:
            for layer in range(1, args.nlayers):
                model.apply_action(-1, layer)
    train_ds = train_ds.with_format("torch")
    test_ds = test_ds.with_format("torch")
    model = model.bfloat16() if 'cuda' in accelerator.device.type else model
    # set up Q-network
    if args.llama:
        args.nlayers = len(model.model.layers)
        print(f"Using {args.nlayers} layers")
    q_net = get_dqn_net(args.nlayers, sum(range(1, args.nlayers + 1)))
    if args.dataset == "squad":
        qlearn_squad_task(model, q_net, train_ds, test_ds, args, accelerator, tokenizer)
    elif args.dataset in ["sst2", "cola", "qnli"] or n_classes == 2:
        q_learn_loop_sst2(model, q_net, train_ds, test_ds, args, accelerator, tokenizer)
    elif n_classes is not None and n_classes > 2:
        q_learn_loop_multiclass(model, q_net, train_ds, test_ds, args, accelerator, tokenizer)
    else:
        q_learn_loop(model, q_net, train_ds, test_ds, args, accelerator, tokenizer)


if __name__ == '__main__':
    arguments = init_args()
    setup_training(arguments)
