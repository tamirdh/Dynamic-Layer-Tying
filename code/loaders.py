from datasets import load_dataset


def preprocess_causal_lm(tokenizer, bptt: int, ds_name: str):
    def tokenization(example):
        key = 'text' if 'text' in example else 'sentence'
        text = "".join(i for i in example[key] if i != '')
        result = tokenizer([text], add_special_tokens=False)
        return result

    block_size = bptt - 1

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        added_tokens = [[tokenizer.bos_token_id] + tokenized_input for tokenized_input in
                        result["input_ids"]]
        result["input_ids"] = added_tokens
        shifted_labels = [ids[1:] + [tokenizer.eos_token_id] for ids in result["input_ids"]]
        result["labels"] = shifted_labels
        result['attention_mask'] = [mask + [1] for mask in result['attention_mask']]
        return result

    if ds_name == "wiki2":
        dataset = load_dataset("wikitext", "wikitext-2-v1")
    elif ds_name == "wiki103":
        dataset = load_dataset("wikitext", "wikitext-103-v1")
    elif ds_name == "lambada":
        dataset = load_dataset("lambada")
    elif ds_name == "ptb":
        dataset = load_dataset("ptb_text_only")
    elif ds_name == "1b":
        dataset = load_dataset("lm1b")
    elif ds_name == "shake":
        dataset = load_dataset('tiny_shakespeare')
    else:
        raise RuntimeError(f"Unrecognized DS name: {ds_name}")

    train_texts, eval_texts = dataset['train'], dataset['test']
    train_encodings = train_texts.map(tokenization, batched=True, remove_columns=train_texts.column_names,
                                      num_proc=8).map(group_texts, batched=True, num_proc=8)
    eval_encodings = eval_texts.map(tokenization, batched=True, remove_columns=eval_texts.column_names,
                                    num_proc=8).map(group_texts, batched=True, num_proc=8)
    return train_encodings, eval_encodings


def preprocess_squad(tokenizer, ds_name: str, bptt: int):
    def tokenization_and_align_labels(examples):
        # Adjustments here
        tokenized_inputs = tokenizer(examples['context'], examples['question'],
                                     truncation=True, padding='max_length', max_length=bptt,
                                     return_offsets_mapping=True)

        start_positions = []
        end_positions = []

        for i in range(len(examples['context'])):
            start_char = examples['answers'][i]['answer_start'][0] if examples['answers'][i]['answer_start'] else 0
            end_char = start_char + len(examples['answers'][i]['text'][0]) if examples['answers'][i]['text'] else 0

            tokenized_context = tokenizer(examples['context'][i], truncation=True, padding='max_length',
                                          max_length=bptt,
                                          return_offsets_mapping=True)

            offsets = tokenized_context.offset_mapping

            start_token_idx = next((idx for idx, offset in enumerate(offsets) if offset[0] <= start_char < offset[1]),
                                   None)
            end_token_idx = next((idx for idx, offset in enumerate(offsets) if offset[0] < end_char <= offset[1]), None)

            # Adjust positions for None values
            if start_token_idx is None or end_token_idx is None or end_token_idx >= bptt:
                context = examples['context'][i]
                new_start = max(0, start_char - (bptt - min(bptt, len(tokenizer.tokenize(examples['question'][i])))))
                new_end = min(len(context),
                              end_char + (bptt - min(bptt, len(tokenizer.tokenize(examples['question'][i])))))
                adjusted_context = context[new_start:new_end]

                tokenized_adjusted = tokenizer(adjusted_context, examples['question'][i],
                                               truncation=True, padding='max_length',
                                               max_length=bptt, return_offsets_mapping=True)

                offsets_adjusted = tokenized_adjusted.offset_mapping
                start_token_idx = next((idx for idx, offset in enumerate(offsets_adjusted) if
                                        offset[0] <= start_char - new_start < offset[1]), None)
                end_token_idx = next((idx for idx, offset in enumerate(offsets_adjusted) if
                                      offset[0] < end_char - new_start <= offset[1]), None)

            start_positions.append(start_token_idx if start_token_idx is not None else tokenizer.cls_token_id)
            end_positions.append(end_token_idx if end_token_idx is not None else tokenizer.cls_token_id)

        tokenized_inputs['start_positions'] = start_positions
        tokenized_inputs['end_positions'] = end_positions
        return tokenized_inputs

    if ds_name == "squad":
        dataset = load_dataset("squad")
    elif ds_name == "squad_v2":
        dataset = load_dataset("squad_v2")
    else:
        raise RuntimeError(f"Unrecognized DS name: {ds_name}")

    train_dataset, eval_dataset = dataset['train'], dataset['validation']
    train_dataset = train_dataset.map(tokenization_and_align_labels, batched=True)
    eval_dataset = eval_dataset.map(tokenization_and_align_labels, batched=True)

    return train_dataset, eval_dataset


def preprocess_glue_sst2(tokenizer, bptt: int, dataset_name: str):
    def tokenization_and_convert_labels(examples):
        # Tokenization
        tokenized_inputs = tokenizer(examples['sentence'], padding='max_length',
                                     truncation=True, max_length=bptt)
        # Convert labels
        tokenized_inputs['labels'] = [label for label in examples['label']]
        return tokenized_inputs

    # Load dataset
    dataset = load_dataset("glue", dataset_name)

    train_dataset, eval_dataset = dataset['train'], dataset['validation']

    # Apply the preprocessing
    train_dataset = train_dataset.map(tokenization_and_convert_labels, batched=True)
    eval_dataset = eval_dataset.map(tokenization_and_convert_labels, batched=True)

    return train_dataset, eval_dataset


def preprocess_glue_single_and_pair(tokenizer, bptt: int, dataset_name: str, pair: bool = False,
                                    keys: tuple = (None, None)):
    def tokenization_and_convert_labels(examples):
        # Tokenization
        if pair:
            tokenized_inputs = tokenizer(examples[keys[0]], examples[keys[1]],
                                         padding='max_length', truncation=True, max_length=bptt)
        else:
            tokenized_inputs = tokenizer(examples['sentence'], padding='max_length',
                                         truncation=True, max_length=bptt)

        # Convert labels
        tokenized_inputs['labels'] = [label for label in examples['label']]
        return tokenized_inputs

    # Load dataset
    dataset = load_dataset("glue", dataset_name)

    train_dataset, eval_dataset = dataset['train'], dataset['test'] if dataset_name!="rte" else dataset['validation']

    # Apply the preprocessing
    train_dataset = train_dataset.map(tokenization_and_convert_labels, batched=True)
    eval_dataset = eval_dataset.map(tokenization_and_convert_labels, batched=True)

    return train_dataset, eval_dataset


def preprocess_images(ds_name: str, image_processor):
    def transform(example):
        inputs = image_processor(example['img'], return_tensors='pt')
        inputs['labels'] = example['label']
        return inputs

    if ds_name == "cifar":
        dataset = load_dataset('cifar10')
    else:
        raise RuntimeError(f"Unrecognized DS name: {ds_name}")
    train_imgs, eval_imgs = dataset['train'], dataset['test']
    train_ds = train_imgs.map(transform, batched=True, remove_columns=train_imgs.column_names, num_proc=1)
    test_ds = eval_imgs.map(transform, batched=True, remove_columns=train_imgs.column_names, num_proc=1)
    return train_ds, test_ds
