local utils = import 'utils.libsonnet';

# No test in quail
# no inputs: "test_race_fmt.json"
# no results: "test_predictions.json"
local common_dependencies = {
  inputs: [
    '${DATA_DIR}/quail/train_race_fmt.json',
    '${DATA_DIR}/quail/dev_race_fmt.json',
  ],
  scripts: [ '${CODE_DIR}/processing/run.sh' ],
  metrics: [ 'train_metrics.json', 'dev_metrics.json' ],
  outputs: [
    'config.json',
    'model.ckpt.data-00000-of-00001',
    'model.ckpt.index',
    'model.ckpt.meta',
    'pytorch_model.bin',
    'vocab.txt',
  ],
  results: [
    'dev_predictions.json',
  ]
};

local common = {
  data_dir: '${DATA_DIR}/quail',
  data_id: 'train_race_fmt.json',
  output_dir: "${MODELS_DIR}/${EXPERIMENT_DIR}",
  output_metrics_dir: "${METRICS_DIR}/${EXPERIMENT_DIR}",
  output_results_dir: "${RESULTS_DIR}/${EXPERIMENT_DIR}",
  model_type: 'bert',
  task_name: 'generic',
  do_train: true,
  do_eval: true,
  fp16: true,
  fp16_opt_level: 'O2',
  loss_scale: 128,
  max_seq_length: 484,
  num_train_epochs: 3,
  per_gpu_train_batch_size: 4,
  per_gpu_eval_batch_size: 16,
  gradient_accumulation_steps: 8,
  learning_rate: 5e-5,
  warmup_proportion: 0.1,
};

local models = {
  bert_quail: {
    output_dir: '${MODEL_DIR}/${EXPERIMENT_DIR}',
    model_name_or_path: '${MODELS_DIR}/bert-base-uncased',
    learning_rate: 5e-5,
  },
  multibert_quail: {
    output_dir: '${MODEL_DIR}/${EXPERIMENT_DIR}',
    model_name_or_path: '${MODELS_DIR}/bert-base-multilingual-cased',
    learning_rate: 5e-5,
  },
};

local modelsTests = [
  item + '-quail-train.json'
  for item in std.objectFields(models)
];

local modelName(testName) = utils.getStringSegment(testName, '-', 0);

local files = {
  [testName]: std.manifestJsonEx({
    inputs: common_dependencies['inputs'] + [models[modelName(testName)]['model_name_or_path']],
    scripts: common_dependencies['scripts'],
    metrics: common_dependencies['metrics'],
    outputs: common_dependencies['outputs'],
    params: common + models[modelName(testName)],
  }, '    ')
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
