local utils = import 'utils.libsonnet';

local common = {
  task_name: 'generic',
  data_dir: 'data/quail',
  model_type: 'bert',
  do_train: true,
  do_eval: true,
  fp16: true,
  fp16_opt_level: 'O2',
  max_seq_length: 484,
  per_gpu_train_batch_size: 8,
  per_gpu_eval_batch_size: 4,
  gradient_accumulation_steps: 4,
  warmup_proportion: 0.1,
  num_train_epochs: 3,
  loss_scale: 128,
};

local models = {
  bert_quail: {
    output_dir: 'data/results/bert-base-uncased-quail',
    model_name_or_path: 'data/models/bert-base-uncased',
    learning_rate: 5e-5,
  },
  multibert_quail: {
    output_dir: 'data/results/bert-base-multilingual-cased-quail',
    model_name_or_path: 'data/models/bert-base-multilingual-cased',
    learning_rate: 5e-5,
  },
};

local modelsTests = [
  item + '-quail-train.json'
  for item in std.objectFields(models)
];

local modelName(testName) = utils.getStringSegment(testName, '-', 0);

local files = {
  [testName]: std.manifestJsonEx(common + models[modelName(testName)], '  ')
  for testName in modelsTests
};

local filelist = {
  'quail-finetuning.filelist': std.join('\n', modelsTests),
};

// object comprehension can only have one item, either filelist or model test files in the export section....
local allFiles = files + filelist;

{
  [fileName]: |||
    %s
  ||| % allFiles[fileName]
  for fileName in std.objectFields(allFiles)
}
