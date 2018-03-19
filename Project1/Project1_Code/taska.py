import helper as hlp
import numpy as np
import matplotlib.pyplot as plt

twenty_train, twenty_test = hlp.fetch_data()
x,y=np.unique(twenty_train.target, return_counts=True)

plt.xlabel("Categories")
plt.ylabel("Number of documents")
plt.title("Histogram of number of documents per category")
plt.bar(x,y)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.xticks(x, hlp.fetch_categories(), rotation='vertical')
plt.show()