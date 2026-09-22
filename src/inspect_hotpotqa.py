from datasets import load_dataset

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor",
    split="validation[:1]",
)

example = dataset[0]

print("ID:")
print(example["id"])

print("\nQUESTION:")
print(example["question"])

print("\nANSWER:")
print(example["answer"])

print("\nSUPPORTING FACTS:")
print(example["supporting_facts"])

print("\nCONTEXT TITLES:")
for title in example["context"]["title"]:
    print(f"- {title}")