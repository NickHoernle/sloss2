from torch import nn


class LogicNet(nn.Module):

    def __init__(self, num_categories):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_categories, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def logic(predictions):
    logic = ((predictions > 0.95) | (predictions < 0.05)).all(dim=1)
    return logic.float()