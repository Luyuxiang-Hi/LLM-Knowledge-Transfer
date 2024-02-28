import torch

def load_checkpoint(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    iltered_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    model.load_state_dict(iltered_state_dict)
    return model