import torch 
import numpy as np

a = torch.rand((3,160,160)) 
b = torch.rand((3,160,160)) 

diff = (a - b)**2
diff = np.array(diff)

summed = np.sum(diff)
err = summed/(160**2)
print(err)
