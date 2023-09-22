import mlconfig
from torch import optim

mlconfig.register(optim.SGD)

mlconfig.register(optim.lr_scheduler.StepLR)
