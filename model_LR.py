#%%
# Linear Regression - The simplest Possible Form

### Part I - Importing Modules and Dataset
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import tensorflow as tf

# # read data
# df = pd.read_csv('./Dataset/salary.csv')
# df.head()

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5*np.pi, 1000)
y = np.sin(x)

plt.plot(x, y)
plt.show()

#%%