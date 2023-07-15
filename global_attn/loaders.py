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
    else:
        raise RuntimeError(f"Unrecognized DS name: {ds_name}")

    train_texts, eval_texts = dataset['train'], dataset['test']
    train_encodings = train_texts.map(tokenization, batched=True, remove_columns=train_texts.column_names,
                                      num_proc=8).map(group_texts, batched=True, num_proc=8)
    eval_encodings = eval_texts.map(tokenization, batched=True, remove_columns=eval_texts.column_names,
                                    num_proc=8).map(group_texts, batched=True, num_proc=8)
    return train_encodings, eval_encodings
