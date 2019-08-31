import os,sys,glob

from denotation.calculate_metric import load_annotations
from collections import defaultdict


def load_main_annotations(
  dir_mainExp = '../AMT/AMTResults/mainExp/train/batches_StoryPlusKeywords2/',
  max_sent = 5):
  print ('Loading main annotations and merge them...',dir_mainExp)

  # load and merge annotations
  annotations = defaultdict(list)
  for fidx, file in enumerate(glob.glob(os.path.join(dir_mainExp,'*.csv'))):
    d_dir = os.path.dirname(file)
    prefix = os.path.basename(file)
    print (fidx, d_dir, prefix)

    annots = load_annotations(d_dir, prefix, max_sent=max_sent)

    # merge this
    for k in annots.keys():
      if k in annotations:
        annotations[k] += annots[k]
      else:
        annotations[k] += annots[k]
    print ('accumulated stories :', len(annotations))
  print ('Done\n')
  return annotations



