import argparse
import json
from pathlib import Path
from datasets import load_dataset


def answer_to_index(answer):
    # Robust mapping: supports 'A'/'B'/'C'/'D' or 0/1/2/3 or 1/2/3/4
    if isinstance(answer, str):
        answer = answer.strip()
        if answer in ["A", "B", "C", "D"]:
            return ord(answer) - ord("A")
        raise ValueError(f"Unsupported string answer format: {answer}")
    if isinstance(answer, int):
        if answer in [0, 1, 2, 3]:
            return answer
        if answer in [1, 2, 3, 4]:
            return answer - 1
    raise ValueError(f"Unsupported answer type/value: {answer} ({type(answer)})")


def build_prompt(question, choices):
    options_lines = [
        f"A. {choices[0]}",
        f"B. {choices[1]}",
        f"C. {choices[2]}",
        f"D. {choices[3]}",
    ]
    options_block = "\n" + "\n".join(options_lines) + "\n"
    prompt = (
        "This is a 4-choice question that you should answer:\n"
        f"{question}{options_block} The correct answer to this question is: "
    )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="mmlu split: test/validation/train")
    parser.add_argument("--subset", type=str, default="all", help="cais/mmlu subset (default: all)")
    parser.add_argument("--out", type=str, default=None, help="output JSONL path")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (data_dir / f"mmlu_{args.subset}_{args.split}.jsonl")

    print(f"Loading dataset: cais/mmlu subset={args.subset} split={args.split}")
    ds = load_dataset("cais/mmlu", args.subset, split=args.split)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            question = ex.get("question")
            choices = ex.get("choices") or ex.get("options")
            answer = ex.get("answer")

            if question is None or choices is None or answer is None:
                raise RuntimeError("Missing fields (question/choices/answer) in the loaded split.")

            if len(choices) != 4:
                raise ValueError(f"Expected 4 choices but got {len(choices)}")

            label_idx = answer_to_index(answer)
            prompt = build_prompt(question, choices)

            obj = {
                "question": question,
                "choices": choices,
                "label_index": int(label_idx),  
                "prompt": prompt,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} examples to {out_path}")


if __name__ == "__main__":
    main()