local utils = import 'utils.libsonnet';

# No test in quail
# no inputs: "test_race_fmt.json"
# no results: "test_predictions.json"
local common_dependencies = {
  inputs: [
    '${DATA_DIR}/quail/dev_race_fmt.json',
  ],
  scripts: [ '${CODE_DIR}/processing/run.sh' ],
  metrics: [ 'dev_metrics.json' ],
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
  data_dir: 'data/quail',
  data_id: 'dev_race_fmt.json',
  output_dir: "${MODELS_DIR}/${EXPERIMENT_DIR}",
  output_metrics_dir: "${METRICS_DIR}/${EXPERIMENT_DIR}",
  output_results_dir: "${RESULTS_DIR}/${EXPERIMENT_DIR}",
  model_type: 'bert',
  task_name: 'generic',
  do_train: false,
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
  overwrite_cache: true,
};

// when not training, model_name_or_path must match the model in output_dir in order
// to load from checkpoint, otherwise will download the model from the internet
local models = {
  bert_race: {
    output_dir: '${MODELS_DIR}/bert-base-uncased-race',
    model_name_or_path: self.output_dir,
  },
  multibert_race: {
    output_dir: '${MODELS_DIR}/bert-base-multilingual-cased-race',
    model_name_or_path: self.output_dir,
  },
  bert_quail: {
    output_dir: '${find_experiment_outputs(bert_quail-quail-train.json)}',
    model_name_or_path: self.output_dir,
  },
  multibert_quail: {
    output_dir: '${find_experiment_outputs(multilingual_quail-quail-train.json)}',
    model_name_or_path: self.output_dir,
  },
};

local testsData = {
  modelNames: std.objectFields(models),
  tests: ['high', 'middle'],
};

# local modelsTests = [
#   item + '.json'
#   for item in utils.generateCombinationsTwoSets(testsData.modelNames, testsData.tests, '%s-%s')
# ];
local modelsTests = [
  item + '-quail-eval.json'
  for item in testsData.modelNames
];

local modelName(testName) = utils.getStringSegment(testName, '-', 0);
local testOnlyName(testName) = utils.getStringSegment(utils.trimExt(testName), '-', 1);

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
  'quail-eval.filelist': std.join('\n', modelsTests),
};

// object comprehension can only have one item, either filelist or model test files in the export section....
local allFiles = files + filelist;

{
  [fileName]: |||
    %s
  ||| % allFiles[fileName]
  for fileName in std.objectFields(allFiles)
}
