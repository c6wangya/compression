import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# data = {
#     'bpp': [0.6953, 
#     0.3267, 
#     0.2856, 
#     0.3349, 
#     0.7364, 
#     0.5268,
#     0.3634,  
#     0.7711,  
#     0.3098,  
#     0.3215,  
#     0.4468,  
#     0.2986,  
#     0.9160,  
#     0.5563,  
#     0.3253,  
#     0.3500,  
#     0.3312,  
#     0.5740, 
#     0.4002, 
#     0.3257, 
#     0.4900, 
#     0.4217, 
#     0.2655, 
#     0.5797],
#     'psnr': [28.74, 
#     31.78, 
#     33.92, 
#     31.99, 
#     28.79, 
#     29.77, 
#     33.13, 
#     27.46, 
#     33.57, 
#     33.15, 
#     30.53, 
#     33.21, 
#     26.17, 
#     29.35, 
#     31.93, 
#     31.95, 
#     32.51, 
#     29.03, 
#     30.87, 
#     32.65, 
#     30.44, 
#     30.53, 
#     34.33, 
#     28.55]
# }
data = {
    'bpp': [0.2983,
    0.4773,
    0.539,
    0.607,
    0.6835, 
    # 0.67,
    # 0.7181,
    0.7323,
    0.8312],
    'psnr': [29.81,
    31.57,
    32.09,
    32.41,
    33.17,
    # 33.51,
    # 33.96,
    33.73, 
    34.55]
}
df = DataFrame(data)
zipped_dict = dict(zip(df['bpp'], df['psnr']))
print(zipped_dict)
zipped_dict = dict(sorted( zipped_dict.items(), key=lambda d: d[0]))
print(zipped_dict)
inn_data = {
    'bpp': [
    0.3076, 
    # 0.3712, 
    0.5125, 
    0.5652, 
    0.6201, 
    # 0.6887, 
    0.6924, 
    0.7332],
    'psnr': [
    30.36, 
    # 30.43, 
    32.18, 
    32.66, 
    33.23, 
    # 33.59, 
    33.59, 
    33.89]
}
inn_df = DataFrame(inn_data)
inn_zipped_dict = dict(zip(inn_df['bpp'], inn_df['psnr']))
print(inn_zipped_dict)
inn_zipped_dict = dict(sorted( inn_zipped_dict.items(), key=lambda d: d[0]))
print(inn_zipped_dict)

inn_data2 = {
    'bpp': [0.3095, 
    # 0.4794, 
    0.5015, 
    # 0.5064, 
    0.5584,
    0.6201, 
    # 0.6887, 
    0.7738, 
    # 0.8811
    ],
    'psnr': [30.47, 
    # 31.92, 
    32.17, 
    # 32.17, 
    32.65, 
    33.3, 
    # 33.59, 
    34.54, 
    # 35.01
    ]
}
inn_df2 = DataFrame(inn_data2)
inn_zipped_dict2 = dict(zip(inn_df2['bpp'], inn_df2['psnr']))
print(inn_zipped_dict2)
inn_zipped_dict2 = dict(sorted( inn_zipped_dict2.items(), key=lambda d: d[0]))
print(inn_zipped_dict2)

plt.plot(list(zipped_dict.keys()), list(zipped_dict.values()), markersize=6, marker='o')
plt.plot(list(inn_zipped_dict.keys()), list(inn_zipped_dict.values()), markersize=6, marker='o')
plt.plot(list(inn_zipped_dict2.keys()), list(inn_zipped_dict2.values()), markersize=6, marker='o')
plt.legend(('baseline', 'inn-fixed-weights', 'inn-adapted-weights'), loc='lower right') 
plt.savefig('plot_psnr.jpg')