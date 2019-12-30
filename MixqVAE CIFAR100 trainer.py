#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from neural import MixqVAE


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Data Loaders

# In[ ]:


BATCH_SIZE = 256
SHUFFLE = True
NUM_WORKERS = 12


# In[ ]:


train_set_loader = data.DataLoader(
    datasets.CIFAR100('./data', train=True, transform=transforms.ToTensor(), download=True),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
)

test_set_loader = data.DataLoader(
    datasets.CIFAR100('./data', train=False, transform=transforms.ToTensor(), download=True),
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
)


# ## Net and Optimizer

# In[ ]:


IN_CHANNELS = 3
NUM_HIDDENS = 256
NUM_RES_HIDDENS = 64
NUM_RES_LAYERS = 4

EMBEDDING_DIM = 128
BETA = 0.9
LMBD = 0.25
NUM_ITER = 3
NUM_EMBEDDINGS = 1024


# In[ ]:


net = MixqVAE(
    IN_CHANNELS,
    NUM_HIDDENS,
    NUM_RES_HIDDENS,
    NUM_RES_LAYERS,
    
    EMBEDDING_DIM,
    BETA,
    LMBD,
    NUM_ITER,
    NUM_EMBEDDINGS,
).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)


# In[ ]:


try:
    net.load_state_dict(torch.load(open('state_dict.pth', 'rb')))
    print('State Dict loaded from \'state_dict.pth\'')
except:
    pass


# ## Train Loop

# In[ ]:


def train(epochs=10):
    print('='*10, end='')
    print(' TRAIN', end=' ') 
    print('='*10, end='\n\n')
    net.train()

    for epoch in range(1, epochs+1):
        running_loss = 0

        for i, batch in enumerate(train_set_loader, 1):
            images, _ = batch
            images = images.to(device)
            
            # Zero grad
            optimizer.zero_grad()

            # Forward
            encoded, quantized, recon_x = net(images)
            # Compute Loss
            loss_value = net.loss_function(images, recon_x, encoded, quantized)
            running_loss += loss_value.item()
            # Backward
            loss_value.backward()
            # Update
            optimizer.step()

            if i % 100 == 0:
                print(f'==> EPOCH[{epoch}]({i}/{len(train_set_loader)}): LOSS: {loss_value.item()}')
            
        print(f'=====> EPOCH[{epoch}] Completed: Avg. LOSS: {running_loss/len(train_set_loader)}')
        print()


# In[ ]:


train(100)


# In[ ]:


image, label = test_set_loader.dataset[10]

encoded, quantized, recon = net(image.unsqueeze(0))

recon = recon[0].squeeze()
plt.imshow(recon.detach().numpy(), cmap='Greys');


# In[ ]:


net.vq.embeddings.weight[0, :]


# In[ ]:


quantized.shape


# In[ ]:


encodings = torch.zeros(28*28, 128)
encodings[14, 12] = 1
encodings[32, 1] = 1
quantized = encodings @ net.vq.embeddings.weight


# In[ ]:


quantized = quantized.reshape((1, 28, 28, 8)).permute(0, 3, 1, 2)


# In[ ]:


recon = net.decoder(quantized)[0].squeeze()
plt.imshow(recon.detach().numpy(), cmap='Greys');


# In[ ]:





# In[ ]:





# In[ ]:


image, label = test_set_loader.dataset[20]
image = image.view(-1, 28*28).unsqueeze(0)

mu, sigma = net.encode(image)
z_2 = mu + sigma * torch.rand_like(sigma)

recon = net.decode(z_2)
print(label)
plt.imshow(recon.view(28, 28).detach().numpy(), cmap='Greys');


# In[ ]:


z_3 = z_2 - z_1
recon = net.decode(z_3)
plt.imshow(recon.view(28, 28).detach().numpy(), cmap='Greys');


# In[ ]:





# In[ ]:





# In[ ]:


torch.save(net.state_dict(), open('vqa_state_dict.pth', 'wb'))


# In[ ]:




