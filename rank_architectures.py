import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default=None, help='Path to file to score')

args = parser.parse_args()

if args.f is not None:
    file = str(args.f)
else:
    file = 'train_history.csv'

if not os.path.exists(file):
    print("Please run `train.py` script to generate architectures first !")
    exit()

lines = []
with open(file, 'r') as f:
    for line in f:
        temp = line.split(',')
        temp[-1] = temp[-1][:-1]  # remove \n
        temp[0] = float(temp[0])  # convert score to float

        # convert the input ids into integers
        for i in range(1, len(temp), 2):
            temp[i] = int(temp[i])

        lines.append(temp)

for i in range(len(lines)):
    lines[i][0] = float(lines[i][0])

points = list(range(len(lines)))
scores = [line[0] for line in lines]
scores = np.array(scores)
std = np.std(scores)

lines = sorted(lines, key=lambda x: x[0], reverse=True)

for line in lines:
    print(line[0], line[1:])

plt.scatter(points, scores)
plt.fill_between(points, scores + std, scores - std, alpha=0.3)
plt.xlabel('Model Index')
plt.ylabel('val-acc')
plt.title('Accuracy over all trained models')
plt.show()