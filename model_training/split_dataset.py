import os
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

output_dir = "data/feedback_qesconv_processed"
os.makedirs(output_dir, exist_ok=True)

# load dataset
ds = load_dataset("avylor/feedback_qesconv")
data = list(ds['train'])

# create train/test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# save splits to processed dir
train_file = os.path.join(output_dir, "train.json")
with open(train_file, 'w') as f:
    json.dump([{
        "text": d["text"],
        "helper_index": d["helper_index"],
        "conv_index": d["conv_index"]
    } for d in train_data], f, indent=4)

test_file = os.path.join(output_dir, "test.json")
with open(test_file, 'w') as f:
    json.dump([{
        "text": d["text"],
        "helper_index": d["helper_index"],
        "conv_index": d["conv_index"]
    } for d in test_data], f, indent=4)

print(f"Created train (80%) split with {len(train_data)} examples in {train_file}")
print(f"Created test (20%) split with {len(test_data)} examples in {test_file}")
