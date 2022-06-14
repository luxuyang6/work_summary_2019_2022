import argparse
import sys
import hashlib
import os

#sys.path.append(os.path.join(os.path.dirname(os.getcwd()),'video_caption_eval_python3','evaluators'))
from evaluators.tokenizer.ptbtokenizer import  PTBTokenizer
from evaluators.bleu.bleu import Bleu
from evaluators.meteor.meteor import Meteor
from evaluators.rouge.rouge import Rouge
from evaluators.cider.cider import Cider
#from spice.spice import Spice


def tokenize(sentence):
  tokenizer = PTBTokenizer()
  return tokenizer.tokenize(sentence)




'''Description:'''

def evaluate(gts,res,metric=None,score_type='macro',tokenized=False):

  if not tokenized:
    '''tokenization...'''
    tokenizer = PTBTokenizer()
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)


  result = []

  if metric is None:
    scorers = [
      (Bleu(4), ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']),
      (Meteor(),'METEOR'),
      (Rouge(), 'ROUGE_L'),
      (Cider(), 'CIDEr')
      #(Spice(), "SPICE")
    ]
  # evaluation on particular numnber of metrics
  else:
    scorers = []
    if 'CIDEr' in metric:
      scorers.append((Cider(),'CIDEr'))
    if 'METEOR' in metric:
      scorers.append((Meteor(),'METEOR'))
    if 'ROUGE_L' in metric:
      scorers.append((Rouge(),'ROUGE_L'))
    if 'BLEU-4' in metric:
      scorers.append((Bleu(4),['BLEU-1','BLEU-2','BLEU-3','BLEU-4']))
    #if 'SPICE' in metric:
    #  scorers.append((Spice(), "SPICE"))

  for scorer, method in scorers:
    score, scores = scorer.compute_score(gts,res)

    if type(method) == list:
      for sc, scs, m in zip(score, scores, method):

        if m == 'BLEU-4':
          if score_type=='macro':
            result.append((m,sc))
          else:
            result.append((m,scs))
    else:
      if score_type == 'macro':
        result.append((method,score))
      else:
        result.append((method,scores))

  if metric is None and score_type=='macro':
        avg = 0.0
        counter = 0
        for method, score in result:
            #if method != 'CIDEr':
            avg += score
            counter += 1 

        result.append(('AVG',avg/counter))

  return result


def test(reference_file,test_file):
  # read from reference file and gererated captions file
  gts = {}
  res = {}

  # load generated captions into a dict

  with open(test_file) as f:
    for line in f:
      line = line.replace('\n','')
      line = line.split('\t')
      assert len(line) == 3
      vid_id = int(int(hashlib.sha256(line[1]).hexdigest(),16) % sys.maxint)
      cap = line[2]
      if res.get(vid_id) is None:
        res[vid_id] = []
        res[vid_id].append(cap)
      else:
        res[vid_id].append(cap)


  # load ground truth captions into a dict
  with open(reference_file) as f:
    for line in f:
      line = line.replace('\n','')
      line = line.split('\t')
      assert len(line) == 2
      vid_id = int(int(hashlib.sha256(line[0]).hexdigest(),16) % sys.maxint)
      cap = line[1]
      if res.get(vid_id) is not None:
        if gts.get(vid_id) is None:
          gts[vid_id] = []
          gts[vid_id].append(cap)
        else:
          gts[vid_id].append(cap)

  assert sorted(gts.keys())==sorted(res.keys())


  metric_scores = evaluate(gts,res,score_type='macro')

  #for method, score in metric_scores:
  #  print method,' : ',score





if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('-ref','--reference_file',default='',help='choose from reference file')
  parser.add_argument('-test','--test_file',default='',help='choose from generated results file')
  args = parser.parse_args()
  test(args.reference_file,args.test_file)

