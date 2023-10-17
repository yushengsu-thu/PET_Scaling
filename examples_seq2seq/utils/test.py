import torch
from torch.optim import AdamW


a = torch.nn.Parameter(torch.tensor([1.0]))
optimizer = AdamW([a], lr=1, weight_decay=0)

while True:
    y = a * a + 3 * a
    # h = y.register_hook(lambda grad: grad.mul_(0.0))

    y.backward()
    a.grad_ = 0
    optimizer.step()
    # h.remove()
    print(a)
