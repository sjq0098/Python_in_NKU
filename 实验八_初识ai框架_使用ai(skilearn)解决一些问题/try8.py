import pandas as pd
import matplotlib.pyplot as plt
student = pd.DataFrame({'sid':[1,2,3,4,5,6], 'sname':['Wang', 'Zhang', 'Li', 'Xu', 'Han', 'Cao'], 'score': [98, 77, 83, 65, 67, 71], 'sclass':[1,1,2,2,1,2]})
print('score>70:\n', student.query('score>70'))
print('score>90 | sclass==2:\n', student.query('score>90 | sclass==2'))

fig, ax = plt.subplots()
ax.set_title('Demo Figure')
ax.set_ylabel('Price')
ax.set_xlabel('Size')
ax.set_xlim(5, 20)
ax.set_ylim(5, 20)
ax.set_xticks([5, 10, 15, 20])
ax.set_xticks(range(5, 21, 1), minor=True)
ax.set_yticks([5, 10, 15, 20])
ax.set_yticks(range(5, 21, 1), minor=True)
plt.plot([6, 8, 10, 14, 18], [7, 9, 13, 17.5, 18])
plt.grid(True)
plt.show()
