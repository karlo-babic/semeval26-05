import torch
import torch.nn as nn
import re
from transformers import AutoModel

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_best_match_span(text, target_word):
    """
    Finds the start and end character indices of the target word in the text.
    Priority: 1. Strict (Word Boundary) -> 2. Loose (Substring) -> 3. Fuzzy (Levenshtein)
    Returns: (start, end) tuple or None
    """
    target_word_esc = re.escape(target_word)
    
    # 1. Strict Match (\bWORD\b)
    match = re.search(r'\b' + target_word_esc + r'\b', text, re.IGNORECASE)
    if match:
        return match.span()

    # 2. Loose Match (Substring)
    match = re.search(target_word_esc, text, re.IGNORECASE)
    if match:
        return match.span()

    # 3. Fuzzy Match (Levenshtein distance on words)
    best_dist = float('inf')
    best_span = None
    target_lower = target_word.lower()
    
    # Iterate through all word-like tokens in the text
    for m in re.finditer(r'\b\w+\b', text):
        word_in_text = m.group()
        dist = levenshtein_distance(word_in_text.lower(), target_lower)
        
        if dist < best_dist:
            best_dist = dist
            best_span = m.span()
            
    return best_span

def inject_meaning(text, target_word, meaning):
    """
    Injects the meaning in brackets immediately after the target word.
    Uses the robust get_best_match_span to find where to insert.
    """
    span = get_best_match_span(text, target_word)
    
    if span:
        _, end = span
        # Insert meaning immediately after the word found
        return text[:end] + f" ({meaning})" + text[end:]
    else:
        # Fallback: Append at the end
        return f"{text} ({meaning})"

class HybridModel_(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Input size is 5 * hidden_size (u, v, |u-v|, u*v, cls)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5)
        )

    def forward(self, input_ids, attention_mask, story_homonym_mask, sense_homonym_mask, token_type_ids=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden = outputs.last_hidden_state # [Batch, Seq, Hidden]

        # 1. Global CLS Embedding
        cls_embedding = last_hidden[:, 0, :]

        # 2. Story Homonym Embedding (u)
        story_mask_expanded = story_homonym_mask.unsqueeze(-1)
        u = torch.sum(last_hidden * story_mask_expanded, dim=1) / torch.clamp(story_mask_expanded.sum(dim=1), min=1e-9)

        # 3. Sense Homonym Embedding (v)
        sense_mask_expanded = sense_homonym_mask.unsqueeze(-1)
        v = torch.sum(last_hidden * sense_mask_expanded, dim=1) / torch.clamp(sense_mask_expanded.sum(dim=1), min=1e-9)

        # 4. Combine Features
        features = torch.cat([u, v, torch.abs(u - v), u * v, cls_embedding], dim=1)
        
        logits = self.ffn(features)
        
        return logits

class HybridModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Input size is 5 * hidden_size (u, v, |u-v|, u*v, cls)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 5)
        )

    def forward(self, input_ids, attention_mask, story_homonym_mask, sense_homonym_mask, token_type_ids=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden = outputs.last_hidden_state # [Batch, Seq, Hidden]

        # 1. Global CLS Embedding
        cls_embedding = last_hidden[:, 0, :]

        # 2. Story Homonym Embedding (u)
        story_mask_expanded = story_homonym_mask.unsqueeze(-1)
        u = torch.sum(last_hidden * story_mask_expanded, dim=1) / torch.clamp(story_mask_expanded.sum(dim=1), min=1e-9)

        # 3. Sense Homonym Embedding (v)
        sense_mask_expanded = sense_homonym_mask.unsqueeze(-1)
        v = torch.sum(last_hidden * sense_mask_expanded, dim=1) / torch.clamp(sense_mask_expanded.sum(dim=1), min=1e-9)

        # 4. Combine Features
        features = torch.cat([u, v, torch.abs(u - v), u * v, cls_embedding], dim=1)
        
        logits = self.ffn(features)
        return logits

class HybridProcessor:
    def __call__(self, tokenizer, sample, max_len):
        homonym = sample['homonym']
        
        story_text = f"{sample['precontext']} {sample['sentence']} {sample['ending']}"
        sense_text = inject_meaning(sample['example_sentence'], homonym, sample['judged_meaning'])

        # Tokenize Pair
        encoding = tokenizer(
            text=story_text, 
            text_pair=sense_text, 
            max_length=max_len, 
            padding='max_length', 
            truncation=True, 
            return_offsets_mapping=True, 
            return_tensors='pt'
        )

        # Identify spans in original strings
        story_span = get_best_match_span(story_text, homonym)
        sense_span = get_best_match_span(sense_text, homonym)

        # Identify token locations
        offsets = encoding['offset_mapping'].squeeze()
        sequence_ids = encoding.sequence_ids() # [None, 0, 0, ..., None, 1, 1, ... None]
        
        story_mask = torch.zeros(max_len)
        sense_mask = torch.zeros(max_len)

        for idx, (ts, te) in enumerate(offsets):
            if ts == te: continue # Skip special tokens
            
            seq_id = sequence_ids[idx]
            
            # Check Story (Sequence 0)
            if seq_id == 0 and story_span:
                if ts >= story_span[0] and te <= story_span[1]:
                    story_mask[idx] = 1.0
            
            # Check Sense (Sequence 1)
            elif seq_id == 1 and sense_span:
                if ts >= sense_span[0] and te <= sense_span[1]:
                    sense_mask[idx] = 1.0

        # Fallback: If truncation cut off the word in either part, point mask to CLS to avoid NaN
        if story_mask.sum() == 0: story_mask[0] = 1.0
        if sense_mask.sum() == 0: sense_mask[0] = 1.0

        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'story_homonym_mask': story_mask,
            'sense_homonym_mask': sense_mask
        }
        
        if 'token_type_ids' in encoding:
            inputs['token_type_ids'] = encoding['token_type_ids'].flatten()
            
        return inputs

ARCHITECTURE_REGISTRY = {
    'hybrid': {
        'model_class': HybridModel,
        'processor': HybridProcessor()
    }
}

def get_architecture(arch_name):
    if arch_name not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHITECTURE_REGISTRY.keys())}")
    return ARCHITECTURE_REGISTRY[arch_name]