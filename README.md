# AdamLRM
Adam optimizer with learning rate multipliers for TensorFlow 2.0.

This Adam optimizer can change learning rate depending on variables to be optimized.
To enable the learning rate multiplier, set a dictionary which has prefixes of variables to apply a multiplier as keys and multipliers as values to `lr_multiplier` of an AdamLRM instance.

```python
from tensorflow import keras
from AdamLRM.adamlrm import AdamLRM

#...

model = keras.Model(inputs=[X], outputs=[Y])

lr_multiplier = {
  'var1':1e-2 # optimize 'var1*' with a smaller learning rate
  'var2':10   # optimize 'var2*' with a larger learning rate
  }
  
opt = AdamLRM(lr=0.001, lr_multiplier=lr_multiplier)

model.compile(
  optimizer=opt,
  loss='mse',
  metrics=['mse'])
```
