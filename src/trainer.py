import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def run_epoch(model, data_loader, optimizer, device, loss_fn=None, is_training=True, aux_weight=0.5):
    """
    Runs one epoch of training or evaluation.
    """
    model.train() if is_training else model.eval()
    total_loss = 0
    
    # For metrics
    predictions_expected_value = [] 
    gold_choices = []
    gold_averages = []

    # Vector of scores [1, 2, 3, 4, 5] to calculate Expected Value
    score_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(device)

    # Keys that are NOT inputs to the model
    non_input_keys = ['id', 'choices', 'target_average', 'target_stdev', 'target_distribution']

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in tqdm(data_loader, desc="Training" if is_training else "Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k not in non_input_keys}
            
            # Forward Pass
            logits = model(**inputs)

            if is_training:
                targets_dist = batch['target_distribution'].to(device)
                
                optimizer.zero_grad()
                
                log_probs = F.log_softmax(logits, dim=1)
                total_batch_loss = loss_fn(log_probs, targets_dist)

                total_batch_loss.backward()
                optimizer.step()
                total_loss += total_batch_loss.item()
            else:
                # Evaluation Logic
                probs = F.softmax(logits, dim=1)
                # Calculate Expected Value (weighted sum of scores)
                batch_expected_values = torch.sum(probs * score_values, dim=1)
                
                predictions_expected_value.extend(batch_expected_values.cpu().numpy())
                gold_choices.extend(batch['choices'])
                gold_averages.extend(batch['target_average'].cpu().numpy())

    if is_training:
        return total_loss / len(data_loader)
    else:
        return predictions_expected_value, gold_choices, gold_averages