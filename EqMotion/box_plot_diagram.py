# plot box plot diagram for LocoVal filtering.
# before_filtering (threshold=0), after_filtering (threshold=0.75, 0.8, 0.85)
# integrate all subset data (eth, hotel, univ, zara1, zara2)

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches

THRESHOLD = [0.8, 0.75, 0]
SUBSET = ['eth', 'hotel', 'univ', 'zara1', 'zara2']

def main():
  data = {0.8: [], 0.75: [], 0: []}

  for threshold in THRESHOLD:
    nums = []
    medians = []
    first_quartiles = []
    third_quartiles = []
    mins = []
    maxs = []
    per_sample = True
    per_subset = not per_sample

    if per_sample:
      for subset in SUBSET:
        with open(os.path.join('output', 'eth', 'box_plot', str(threshold), f'{subset}.json'), 'r') as f:
          boxplot_dict = json.load(f)
          print('path:', os.path.join('output', 'eth', 'box_plot', str(threshold), f'{subset}.json'))
          print('boxplot_dict:', boxplot_dict)
          num_samples = boxplot_dict['num_samples']
          medians.append(boxplot_dict['median_mean'])
          first_quartiles.append(boxplot_dict['1q_mean'])
          third_quartiles.append(boxplot_dict['3q_mean'])
          mins.append(boxplot_dict['min_mean'])
          maxs.append(boxplot_dict['max_mean'])
          nums.append(num_samples)

      # each sample equally contributes to the mean
      data[threshold].append(sum(mins) / sum(nums))
      data[threshold].append(sum(first_quartiles) / sum(nums))
      data[threshold].append(sum(medians) / sum(nums))
      data[threshold].append(sum(third_quartiles) / sum(nums))
      data[threshold].append(sum(maxs) / sum(nums))

    if per_subset:
      for subset in SUBSET:
        with open(os.path.join('output', 'eth', 'box_plot', str(threshold), f'{subset}.json'), 'r') as f:
          boxplot_dict = json.load(f)
          print('path:', os.path.join('output', 'eth', 'box_plot', str(threshold), f'{subset}.json'))
          print('boxplot_dict:', boxplot_dict)
          num_samples = boxplot_dict['num_samples']
          medians.append(boxplot_dict['median_mean']/num_samples)
          first_quartiles.append(boxplot_dict['1q_mean']/num_samples)
          third_quartiles.append(boxplot_dict['3q_mean']/num_samples)
          mins.append(boxplot_dict['min_mean']/num_samples)
          maxs.append(boxplot_dict['max_mean']/num_samples)

      # each subset equally contributes to the mean
      data[threshold].append(sum(mins)/len(SUBSET))
      data[threshold].append(sum(first_quartiles)/len(SUBSET))
      data[threshold].append(sum(medians)/len(SUBSET))
      data[threshold].append(sum(third_quartiles)/len(SUBSET))
      data[threshold].append(sum(maxs)/len(SUBSET))


  # plot box plot diagram for each threshold
  x_labels = ['λ=0.8', 'λ=0.75', 'w/o filtering']
  y_ticks = [0.2, 0.5, 1, 1.5, 2, 5, 10, 20, ]
  y_labels = [str(y) for y in y_ticks]
  figsize = (7, 2) # height: 4, width: 6

  fig, ax = plt.subplots(figsize=figsize, layout='tight')
  bp = ax.boxplot(data.values(), whis=(0, 100), vert=False, tick_labels=x_labels)
  colors=['salmon', 'skyblue', 'palegreen']
  borders = ['red', 'blue', 'green']

  for artist, color, border in zip(bp['boxes'], colors, borders):
    patch = mpatches.PathPatch(artist.get_path(), color=color)
    ax.add_artist(patch)
    artist.set_color(border)

  for median in bp['medians']:
    median.set_color('black')

  ax.set_ylabel('Threshold', fontsize=16)
  ax.set_xlabel('ADE', fontsize=16)
  # ax.set_title('Box plot diagram for LocoVal filtering')
  ax.set_xscale('log')
  ax.set_xticks(y_ticks)
  ax.set_yticks([])
  ax.set_xticklabels(y_labels, fontsize=12)

  ax.legend(x_labels, fontsize=12)

  plt.savefig('box_plot_diagram_log.png')
  plt.savefig('box_plot_diagram_log.pdf', bbox_inches='tight')

if __name__ == '__main__':
  main()