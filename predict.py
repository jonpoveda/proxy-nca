import torch

import net

from utils import evaluate

model = net.bn_inception(pretrained=True)
net.embed(model, sz_embedding=64)

model_path = "model/{}.pt".format('juanito.pt')
model.load_state_dict(torch.load(model_path))
model.eval()

dataloader = None
evaluate(model, dataloader)
