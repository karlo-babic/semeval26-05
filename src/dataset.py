import json
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

class AmbiStoryDataset(Dataset):
    def __init__(self, file_path, tokenizer, processor, max_len=256):
        """
        Args:
            file_path (str): Path to json data.
            tokenizer: HF Tokenizer.
            processor: Processor function/class from architecture.
            max_len (int): Max sequence length.
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len = max_len
        
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        
        self.samples = []
        for sample_id, content in self.data.items():
            content['id'] = sample_id
            self.samples.append(content)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Processor handles text
        inputs = self.processor(self.tokenizer, sample, self.max_len)

        # --- Calculate Label Distribution ---
        # Create a zero tensor for 5 classes (scores 1, 2, 3, 4, 5)
        distribution = torch.zeros(5, dtype=torch.float)
        
        # Safely get fields that might be missing or placeholders in test set
        raw_choices = sample.get('choices', [])
        raw_avg = sample.get('average', 0.0)
        raw_stdev = sample.get('stdev', 0.0)
        
        # 1. Handle Choices
        # In test set, choices might be "(???)" or similar strings.
        valid_choices = []
        if isinstance(raw_choices, list):
            for c in raw_choices:
                if isinstance(c, (int, float)) and 1 <= c <= 5:
                    distribution[int(c)-1] += 1
                    valid_choices.append(c)
            
            # Normalize
            if distribution.sum() > 0:
                distribution = distribution / distribution.sum()
        else:
            # If choices is "(???)" or None, treat as empty
            valid_choices = []

        # 2. Handle Average & Stdev
        # Ensure they are floats; otherwise default to 0.0
        try:
            avg = float(raw_avg)
        except (ValueError, TypeError):
            avg = 0.0
            
        try:
            stdev = float(raw_stdev)
        except (ValueError, TypeError):
            stdev = 0.0

        targets = {
            'id': sample['id'],
            'target_average': torch.tensor(avg, dtype=torch.float),
            'target_stdev': torch.tensor(stdev, dtype=torch.float),
            'target_distribution': distribution,
            'choices': valid_choices 
        }
        
        return {**inputs, **targets}

def custom_collate_fn(batch):
    choices_list = [item.pop('choices') for item in batch]
    ids_list = [item.pop('id') for item in batch]
    collated_batch = default_collate(batch)
    collated_batch['choices'] = choices_list
    collated_batch['id'] = ids_list
    return collated_batch