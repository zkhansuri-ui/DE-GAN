import os
import random

dataset_path = 'path/to/data' 
config_dir = 'path/to/data' 

train_output_path = os.path.join(config_dir, 'train_name_all.txt')
val_output_path = os.path.join(config_dir, 'val_name_all.txt')
test_output_path = os.path.join(config_dir, 'test_name_all.txt')

names = []
if os.path.exists(dataset_path):
    names = sorted([f for f in os.listdir(dataset_path) if f.startswith('BraTS')])

# Ensure reproducibility
random.seed(42)
random.shuffle(names)

total = len(names)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_names = names[:train_end]
val_names = names[train_end:val_end]
test_names = names[val_end:]

def write_names(path, name_list):
    with open(path, 'w') as f:
        for name in name_list:
            f.write(f'{name}\n')
    print(f'Successfully wrote {len(name_list)} patient names to {path}')

write_names(train_output_path, train_names)
write_names(val_output_path, val_names)
write_names(test_output_path, test_names)
