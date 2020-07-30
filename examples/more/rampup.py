"""

"""


from thexp.calculate.rampup import *


from matplotlib import  pyplot as plt


plt.plot([linear_rampup(i, 400, 1.4, end=2.6) for i in range(400)])
plt.show()
# plt.plot([linear_rampup(i, 100, 0.5, end=2) for i in range(200)])
# plt.show()
# plt.plot([sigmoid_rampup(i, 100, 0.5) for i in range(200)])
# plt.show()
