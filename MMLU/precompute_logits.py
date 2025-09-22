import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.utils.data as tdata
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_jsonl(path):
    prompts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
            labels.append(int(obj["label_index"]))
    return prompts, labels


def build_model_and_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        load_in_4bit=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,  
    )
    model.eval()
    model.config.pad_token_id = tok.pad_token_id
    return tok, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSONL produced by process_mmlu.py")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-name", type=str, default="llama31-8b-instruct_mmlu_test",
                        help="basename for logits_cache/<out-name>_logits.pkl")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device_for_inputs = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading prompts from: {args.input}")
    prompts, labels = load_jsonl(args.input)
    n = len(prompts)
    print(f"Total examples: {n}")

    print(f"Loading model (8-bit, fp16 compute): {args.model}")
    tok, model = build_model_and_tokenizer(args.model)

    letters = ["A", "B", "C", "D"]
    letter_ids = []
    for ch in letters:
        ids = tok.encode(ch, add_special_tokens=False)
        if len(ids) == 1:
            letter_ids.append(ids[0])
        else:
            ids2 = tok.encode(" " + ch, add_special_tokens=False)
            if len(ids2) == 1:
                letter_ids.append(ids2[0])
            else:
                raise ValueError(f"Letter {ch} is not a single token: {ids} / {ids2}")
    letter_ids_tensor = torch.tensor(letter_ids, dtype=torch.long) 

    all_logits = []
    all_reps = []
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]
            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tok.model_max_length,
            )
            input_ids = enc["input_ids"].to(device_for_inputs)
            attn = enc["attention_mask"].to(device_for_inputs)

            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            last_idx = attn.sum(dim=1) - 1  # [bsz]
            bsz = input_ids.size(0)
            logits_last = out.logits[torch.arange(bsz, device=last_idx.device), last_idx, :]  # [bsz, vocab]

            idx = letter_ids_tensor.to(logits_last.device)
            selected_logits = logits_last.index_select(dim=1, index=idx).to(torch.float32)  # [bsz, 4]
            all_logits.append(selected_logits.detach().cpu())

            last_hidden = out.hidden_states[-1]  # [bsz, seq, hidden]
            reps_last = last_hidden[torch.arange(bsz, device=last_idx.device), last_idx, :].to(torch.float32)  # [bsz, H]
            all_reps.append(reps_last.detach().cpu())

            if (i // args.batch_size) % 20 == 0:
                print(f"Processed {min(i + bsz, n)}/{n}")

    logits_tensor = torch.cat(all_logits, dim=0)  # [N, 4]
    reps_tensor = torch.cat(all_reps, dim=0)      # [N, H]
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # [N]

    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "logits_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_logits.pkl"
    out_path_reps = out_dir / f"{args.out_name}_reps.pkl"

    dataset = tdata.TensorDataset(logits_tensor, labels_tensor)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved logits and targets to: {out_path}")
    print(f"Tensor shapes: logits={tuple(logits_tensor.shape)}, targets={tuple(labels_tensor.shape)}")

    reps_dataset = tdata.TensorDataset(reps_tensor, labels_tensor)
    with open(out_path_reps, "wb") as f:
        pickle.dump(reps_dataset, f)

    print(f"Saved last-layer reps and targets to: {out_path_reps}")
    print(f"Tensor shapes: reps={tuple(reps_tensor.shape)}, targets={tuple(labels_tensor.shape)}")

    predictions = torch.argmax(logits_tensor, dim=1)
    accuracy = (predictions == labels_tensor).float().mean()
    print(f"Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()