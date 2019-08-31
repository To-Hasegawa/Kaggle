# -*- coding: utf-8 -*-
#Show memory usage by each columns

print('Shape: ', df.shape)
print('Total: ', df.memory_usage().sum())
df.memory_usage()