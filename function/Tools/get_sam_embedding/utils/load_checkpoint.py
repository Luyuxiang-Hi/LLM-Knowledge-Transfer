import torch

def load_checkpoint(model, checkpoint):
    state_dict = torch.load(checkpoint)
    image_encoder_state_dict = {k.replace('image_encoder.', ''): v for k, v in state_dict.items() if k.startswith('image_encoder.')}
    model.load_state_dict(image_encoder_state_dict)
    return model