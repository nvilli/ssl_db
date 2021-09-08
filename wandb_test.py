import wandb
import random
wandb.init(project="test")

wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128

loss = 0
for epoch in range(10):
    loss += 1
    wandb.log({'epoch': epoch, 'loss': loss})

wandb.save("mymodel.h5")