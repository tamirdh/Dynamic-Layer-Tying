from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split


def preprocess_causal_lm(tokenizer, bptt:int, ds_name:str):
    def tokenization(example):
        key = 'text' if 'text' in example else 'sentence'
        text = "".join(i for i in example[key] if i != '')
        result = tokenizer([text], add_special_tokens=False)
        return result

    block_size = bptt -1

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
    else:
        raise RuntimeError(f"Unrecognized DS name: {ds_name}")

    train_texts, eval_texts = dataset['train'], dataset['test']
    train_encodings = train_texts.map(tokenization, batched=True, remove_columns=train_texts.column_names,
                                      num_proc=8).map(group_texts, batched=True, num_proc=8)
    eval_encodings = eval_texts.map(tokenization, batched=True, remove_columns=eval_texts.column_names,
                                    num_proc=8).map(group_texts, batched=True, num_proc=8)
    return train_encodings, eval_encodings


def preprocess_squad(tokenizer):
    squad = load_dataset("squad")
    max_length = 2048
    stride = 50

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) - 1 and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    train_ds = squad["train"]
    test_ds = squad['validation']
    train_ds = train_ds.map(preprocess_training_examples, batched=True, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(preprocess_validation_examples, batched=True, remove_columns=test_ds.column_names)
    return train_ds, test_ds
