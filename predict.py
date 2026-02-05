import json
import torch
import argparse
import yaml
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.dataset import AmbiStoryDataset, custom_collate_fn
from src.architecture import get_architecture

def predict(model, data_loader, device):
    model.eval()
    results = []

    # Vector of scores [1, 2, 3, 4, 5] for Expected Value calculation
    score_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device)
    
    non_input_keys = ['id', 'choices', 'target_average', 'target_stdev', 'target_distribution']

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in batch.items() if k not in non_input_keys}
            ids = batch['id']

            outputs = model(**inputs)
            
            # Extract logits if output is a dictionary
            if isinstance(outputs, dict):
                outputs = outputs["logits"]

            # --- DYNAMICALLY HANDLE OUTPUT ---
            # If output is regression (shape [B, 1] or [B])
            if outputs.ndim <= 1 or outputs.shape[1] == 1:
                preds = torch.clamp(outputs.squeeze(), 1, 5)
            # If output is distribution (shape [B, 5])
            else:
                probs = F.softmax(outputs, dim=1)
                # Calculate Expected Value
                preds = torch.sum(probs * score_values, dim=1)
            
            #final_preds = torch.round(preds).int()
            #final_preds = torch.clamp(final_preds, 1, 5)
            final_preds = torch.clamp(preds, 1.0, 5.0)

            # Handle single item batch edge case
            if final_preds.ndim == 0:
                final_preds = final_preds.unsqueeze(0)

            for i in range(len(ids)):
                results.append({"id": ids[i], "prediction": final_preds[i].item()})

    return results

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Architecture Setup ---
    arch_name = config['model']['architecture']
    model_name = config['model']['name']
    arch_config = get_architecture(arch_name)
    ModelClass = arch_config['model_class']
    input_processor = arch_config['processor']
    
    print(f"Loading {arch_name} model from {args.checkpoint_path}...")
    model = ModelClass(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)

    # --- Data Loading ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = AmbiStoryDataset(args.input_file, tokenizer, input_processor, config['training']['max_len'])
    
    # Use batch_size from config for consistency, but allow override
    batch_size = config['training'].get('batch_size', 8)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # --- Prediction ---
    predictions = predict(model, loader, device)
    
    # Sort by ID to ensure submission order is correct
    predictions.sort(key=lambda x: int(x['id']))

    # --- Save Results ---
    with open(args.output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Predictions saved to {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the training config used (to determine architecture)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the best_model.pt checkpoint')
    parser.add_argument('--input_file', type=str, default='data/dev.json', help='Path to input data (dev.json or test.json)')
    parser.add_argument('--output_file', type=str, default='input/res/predictions.jsonl', help='Output path for predictions.jsonl')
    args = parser.parse_args()
    main(args)