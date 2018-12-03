import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

# data reading
data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()

# collecting X and Y
x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values

# mean of x and y
mx = np.mean(x)
my = np.mean(y)

# total number of values
l = len(x)

# using formula to calculate m and c
num = 0
den = 0
for i in range(l):
    num += (x[i] - mx)*(y[i] - my)
    den += (x[i] - mx)**2

m = num / den
c = my - (m * mx)

# r_square method for error
ss_t = 0
ss_r = 0
for k in range(l):
    yp = c + m * x[k]
    ss_t += (y[k] - my)**2
    ss_r += (y[k] - yp)**2
r2 = (ss_r/ss_t) / l
print("Slope(m) = {0}  Constant(c) = {1}  Error(R²) = {2}".format(m, c, r2))

# plotting values and regression line
max_x = np.max(x) + 100
min_x = np.min(x) - 100

# calculating line values x and y
x_axis = np.linspace(min_x, max_x, 1000)
y_axis = m * x_axis + c + r2

# plotting line
plt.plot(x_axis, y_axis, color='k', label='Regression Line')

# scatter point
plt.scatter(x, y, color='b', label='Scatter Point')

plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


