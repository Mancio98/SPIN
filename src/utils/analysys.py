import numpy as np
import torch

@torch.no_grad
def get_features_embedding(model, loader, device, method='mean'):

    if next(model.parameters()).device != device:
        model = model.to(device)

    model.eval()

    features_vectors = []
    targets_list = []
    for data_map, targets in loader:
       
        inputs = {k: v.to(device) for k, v in data_map.items()}

        outputs = model(**inputs, output_hidden_states=True)

        feat = outputs.hidden_states[-1].cpu().numpy()

        if method == 'mean':

            filtered_sequence = np.full(feat.shape, np.NaN)
            # Copy the values from b to filtered_b where a is True
            input_ids = inputs['input_ids'].cpu().numpy()
            only_sequence_mask = (input_ids >= 4) & (input_ids < 29)
            filtered_sequence[only_sequence_mask] = feat[only_sequence_mask]

            features_vectors.extend(np.nanmean(filtered_sequence, axis=1))
            
        elif method == 'cls':
            features_vectors.extend(feat[:,0])
        
        targets_list.extend(targets[:,0].numpy())

    return np.array(features_vectors), np.array(targets_list)