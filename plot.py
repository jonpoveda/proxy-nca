from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

dirpath = Path(__file__).parent.joinpath('log')

paths = [
    dirpath.joinpath('losses.txt'),
    dirpath.joinpath('nmis.txt'),
    dirpath.joinpath('recalls1.txt'),
    dirpath.joinpath('recalls2.txt'),
    dirpath.joinpath('recalls4.txt'),
    dirpath.joinpath('recalls8.txt')
]

with dirpath.joinpath('epochs.txt').open('r') as file:
    rows = file.readlines()
    rows = [float(e.strip()) for e in rows]
    rows = np.array(rows)
    dalog = rows

for path in paths:
    print('Readading {}'.format(path), flush=True)
    with path.open('r') as file:
        rows = file.readlines()
        rows = [float(e.strip()) for e in rows]
        rows = np.array(rows)
        dalog = np.vstack((dalog, rows))

dalog = dalog.T

# Remove non-used data (21:71 rows: epochs 20-69)
chunk1 = dalog[0:21, :]
chunk2 = dalog[71:, :]
dalog = np.concatenate((chunk1, chunk2))
assert np.max(dalog[:, 0]) == 19.

# plt.figure()
# plt.plot(dalog[:, 0])
# plt.title('Epochs')

plt.figure()
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(dalog[:, 1])
plt.autoscale(axis='both', tight=True)
plt.savefig('Loss.png', bbox_inches='tight')
# plt.show()

plt.figure()
plt.title('NMI')
plt.xlabel('epoch')
plt.ylabel('NMI (%)')
plt.plot(dalog[:, 2])
plt.autoscale(axis='x', tight=True)
plt.savefig('NMI.png', bbox_inches='tight')
# plt.show()

plt.figure()
plt.title('R@k')
plt.xlabel('epoch')
plt.ylabel('Recall (%)')
plt.plot(dalog[:, 3])
plt.plot(dalog[:, 4])
plt.plot(dalog[:, 5])
plt.plot(dalog[:, 6])
plt.autoscale(axis='x', tight=True)
plt.ylim((96, 101))
plt.legend(('R@1', 'R@2', 'R@4', 'R@8',))
plt.savefig('R_at_k.png', bbox_inches='tight')
# plt.show()

plt.figure()
plt.title('R@1')
plt.xlabel('epoch')
plt.ylabel('Recall (%)')
plt.plot(dalog[:, 3])
plt.autoscale(axis='x', tight=True)
plt.ylim((96, 101))
plt.savefig('R_at_1.png', bbox_inches='tight')
# plt.show()
