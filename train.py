import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from scipy.stats import spearmanr

from src.dataset import AmbiStoryDataset, custom_collate_fn
from src.architecture import get_architecture
from src.trainer import run_epoch
from evaluate import is_within_standard_deviation

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Directories ---
    output_dir = config['training']['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Architecture Setup ---
    arch_name = config['model']['architecture']
    print(f" initializing {arch_name}...")
    
    # Get the specific classes from registry
    arch_config = get_architecture(arch_name)
    ModelClass = arch_config['model_class']
    input_processor = arch_config['processor']

    # --- Data & Model ---
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    print("Loading Train Data...")
    train_ds = AmbiStoryDataset('data/train.json', tokenizer, input_processor, config['training']['max_len'])

    print("Loading Dev Data...")
    dev_ds = AmbiStoryDataset('data/dev.json', tokenizer, input_processor, config['training']['max_len'])

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    model = ModelClass(config['model']['name']).to(device)

    # ============================================================
    # OPTIMIZER SETUP (Freezing + Differential LR)
    # ============================================================
    
    # Handle Freezing (Optional)
    if config['training'].get('freeze_backbone', False):
        print("--> Freezing Backbone Weights")
        for name, param in model.named_parameters():
            # We freeze everything EXCEPT the head (ffn/classifier)
            if "ffn" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Define Learning Rates
    base_lr = float(config['training']['learning_rate'])
    head_lr = base_lr * 10  # Head learns 10x faster
    
    # Create Parameter Groups
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if ("ffn" in n or "classifier" in n) and p.requires_grad
            ],
            "lr": head_lr,
        },
        {
            "params": [
                p for n, p in model.named_parameters() 
                if not ("ffn" in n or "classifier" in n) and p.requires_grad
            ],
            "lr": base_lr,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    # KLDivLoss with 'batchmean' (for distribution learning)
    loss_fn = nn.KLDivLoss(reduction='batchmean') 

    best_accuracy = 0.0

    for epoch in range(config['training']['epochs']):
        print(f"\n--- Epoch {epoch + 1} ---")
        
        train_loss = run_epoch(model, train_loader, optimizer, device, loss_fn, is_training=True)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluation
        preds, choices, avgs = run_epoch(model, dev_loader, None, device, is_training=False)

        # Metrics
        spearman = spearmanr(preds, avgs)[0]
        
        # Accuracy calc (rounding the Expected Value)
        int_preds = [max(1, min(5, round(p))) for p in preds]
        correct = sum([1 for i, p in enumerate(int_preds) if is_within_standard_deviation(p, choices[i])])
        accuracy = correct / len(int_preds)

        print(f"Spearman: {spearman:.4f} | Accuracy: {accuracy:.4f}")

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
            print("-> Saved Best Model")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)