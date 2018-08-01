import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./pima-data.csv')
print(df.shape)
