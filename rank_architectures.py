import os
import glob
import argparse
import numpy as np
import warnings
import matplotlib.pyplot as plt

try:
    import mplcursors
    _HAS_MPLCURSORS = True

except ImportError:
    _HAS_MPLCURSORS = False

    warnings.warn('Package `mplcursors` not found. This is useful for visualizing the '
                  'configuration string when hovering over the plotted data. Use '
                  '`pip install mplcursors` to obtain the package.')

plt.style.use('seaborn-paper')

parser = argparse.ArgumentParser(description='Plots the model architecture and it\'s respective '
                                             'score (whether obtained via training or predicted '
                                             'by the Controller RNN).')

parser.add_argument('-f', type=str, default=None, nargs='+', help='List of paths to files which will be scores. '
                                                                  'Defaults to the train_history.csv file.')
parser.add_argument('-sort', dest='sort', action='store_true')

parser.set_defaults(sort=False)

args = parser.parse_args()

if args.f is not None:
    fns = [str(fn) for fn in args.f]

    files = []
    for f in fns:
        if '*' in f:
            f = sorted(list(glob.glob(f)))
        else:
            f = [f]

        files.extend(f)

    # always place train_history at the start
    if 'train_history.csv' in files:
        files.remove('train_history.csv')
        files.insert(0, 'train_history.csv')

else:
    files = ['train_history.csv']

# check all the files in the list exist
for file in files:
    if not os.path.exists(file):
        print("Could not find file at path : %s !" % (file))
        print("Please run `train.py` script to generate architectures first or "
              "run `score_architectures.py` to obtain the predicted accuracy of "
              "children model combinations!")
        exit()

# read the file data and extract score and model architecture
lines = []
for file in files:
    with open(file, 'r') as f:
        for line in f:
            temp = line.split(',')
            temp[-1] = temp[-1][:-1]  # remove \n
            temp[0] = float(temp[0])  # convert score to float

            # convert the input ids into integers
            for i in range(1, len(temp), 2):
                temp[i] = int(temp[i])

            lines.append(temp)

# convert the score to floating point
for i in range(len(lines)):
    lines[i][0] = float(lines[i][0])

# prepare for plotting
points = list(range(len(lines)))

# normalize scores to color the points when plotting
sorted_lines = sorted(lines, key=lambda x: x[0], reverse=True)
max_score = float(sorted_lines[0][0])
min_score = float(sorted_lines[-1][0])

if args.sort:
    sorted_lines.reverse()
    lines = sorted_lines

scores = [line[0] for line in lines]
scores = np.array(scores)
normalized_scores = (scores - min_score) / (max_score - min_score)

plt.scatter(points, scores, c=normalized_scores, cmap='coolwarm')
plt.xlabel('Model Index')
plt.ylabel('Accuracy')
plt.title('Model Definition vs Accuracy')

# annotate the points with the architecture details upon hovering
if _HAS_MPLCURSORS:
    cursor = mplcursors.cursor(hover=True)
    cursor.connect('add', lambda sel: sel.annotation.set_text(lines[sel.target.index][1:]))

plt.show()