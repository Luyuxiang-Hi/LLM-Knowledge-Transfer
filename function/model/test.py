from deeplabv3.deeplabv3 import Net
if __name__ == '__main__':
    model = Net()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: ", total_params)