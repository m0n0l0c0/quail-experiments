#!/usr/bin/env python
import sys
import json

def json_args_to_array_args(json_args):
  array_args = []
  for arg_key, arg_value in json_args.items():
    array_args.append('--' + arg_key)
    if type(arg_value) is bool:
      continue
    # should be stringified?
    array_args.append(str(arg_value))
  return array_args

def main(json_file):
  json_args = json.load(open(json_file, 'r'))
  array_args = json_args_to_array_args(json_args)
  print(' '.join(array_args))

if __name__ == "__main__":
  if len(sys.argv) == 2:
    main(sys.argv[1])
