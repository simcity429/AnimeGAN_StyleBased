import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class tmp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(2500, 100)
        self.l2 = torch.nn.Linear(100,100)
        self.l3 = torch.nn.Linear(100,100)
        self.l4 = torch.nn.Linear(100,100)
        self.l5 = torch.nn.Linear(100,100)
        self.l6 = torch.nn.Linear(100,100)
        self.l7 = torch.nn.Linear(100,100)
        self.l8 = torch.nn.Linear(100,100)
        self.l9 = torch.nn.Linear(100,100)
        self.l10 = torch.nn.Linear(100,100)
        self.l11 = torch.nn.Linear(100,62)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        return x


m = tmp()
x = torch.rand(10, 2500)
m(x)
print(count_parameters(m))


