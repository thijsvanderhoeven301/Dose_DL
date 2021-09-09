import torch
import numpy as np
import config

# def save_some_examples(gen, val_loader, epoch, folder):
#     x, y = next(iter(val_loader))
#     x, y = x.to(config.DEVICE), y.to(config.DEVICE)
#     gen.eval()
#     with torch.no_grad():
#         y_fake = gen(x)
#         np.save(folder + f"/y_gen_{epoch}.npy", y_fake.numpy())
#         np.save(folder + f"/input_{epoch}.npy", x.numpy())
#         if epoch == 1:
#             np.save(folder + f"/label_{epoch}.png")
#     gen.train()

def save_checkpoint(model, optimizer, filename = "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint of {model}".format(model = model.__class__.__name__))
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint of {model}".format(model = model.__class__.__name__))
    checkpoint = torch.load(checkpoint_file, map_loaction = config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr