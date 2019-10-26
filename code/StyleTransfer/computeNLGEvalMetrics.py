from nlgeval import compute_metrics
import sys
import os

metrics_dist = compute_metrics(hypothesis=sys.argv[1],references=[sys.argv[2],],no_skipthoughts=True)

#BLEUOutput = os.popen("perl multi-bleu.perl -lc "+sys.argv[2]+" < "+sys.argv[1]).read()
BLEU2Output = os.popen("perl BLEU2.perl -lc "+sys.argv[2]+" < "+sys.argv[1]).read()
 
print(BLEUOutput)
print(BLEU2Output)
