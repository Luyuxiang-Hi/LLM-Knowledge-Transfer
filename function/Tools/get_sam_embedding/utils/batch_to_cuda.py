def batch_to_cuda(batch, gpu):
    # Send data to computing device:
    for key, item in batch.items():
        if hasattr(item, "cuda"):
            batch[key] = item.cuda(gpu) # non_blocking=False
    return batch