#!/usr/bin/env python

import sys
import json
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('data', metavar='data.json',
      help='Data to convert to dataset')
  parser.add_argument('answers', metavar='answers.json',
      help='Answers to store gold standard in the dataset')
  parser.add_argument('-o', '--output', default=None,
      help='Output file to write dataset (default stdout)')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  return parser.parse_args()

def merge_data_with_labels(data, gold_answers):
  dataset = []
  for id, item in data.items():
    questions, options, answers = [], [], []
    for q_id, question_data in item['questions'].items():
      # gold_answers
      #   "f001_0": "3", ...
      # race format:
      #   "C"
      questions.append(question_data['question'])
      options.append(list(question_data['answers'].values()))
      answers.append(chr(ord('A') + int(gold_answers[q_id])))

    assert(len(questions) == len(options) == len(answers))
    example = dict(id=id, questions=questions, options=options, answers=answers)
    dataset.append(example)

  return dataset

def main(args):
  questions = json.load(open(args.data, 'r'))
  answers = json.load(open(args.answers, 'r'))
  dataset = merge_data_with_labels(questions['data'], answers['data'])
  data = dict(version=questions['version'], data=dataset)
  data_print = json.dumps(obj=data, ensure_ascii=False) + '\n'
  if args.output is None:
    print(data)
  else:
    with open(args.output, 'w') as fstream:
      fstream.write(data_print)

if __name__ == '__main__':
  args = parse_args()
  main(args)

