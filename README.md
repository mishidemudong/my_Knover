# Knover
Knover is a toolkit for knowledge grounded dialogue generation based on PaddlePaddle. Knover allows researchers and developers to carry out efficient training/inference of large-scale dialogue generation models. 

### What's New:

- July 2020: We are opening [PLATO-2](plato-2/README.md), a large-scale generative model with latent space for open-domain dialogue systems.


## Basic usage:

### Training
Carry out local training with a configuration file. You can specify GPU by `export CUDA_VISIBLE_DEVICES=XXX` in `./scripts/local/train.sh`. You can specify other environment variables in the script.

``` bash
./scripts/local/train.sh ${TRAIN_CONF}
```

An example of training configuration files is `./package/dialog_en/plato/24L_train.conf`. It contains three sections: `job`, `task` and `training`.

#### job

This section defines:

`job_script`: the main script of this task; use `./scripts/distributed/train.sh` for training task.

#### task
This section defines:

`model`: the used model class

`task`: task name

`vocab_path`: vocabulary path

tokenizer related: `spm_model_file` for SentencePieces Tokenizer, and so on.

dataset files related: `train_file`, `valid_file`, `data_format` and `file_format`.

`config_path`: model configuration file.

Choices of `data_format`:

- `raw`: untokenized data tsv file, example: `./data/train.tsv`, each column is a field.

- `tokenized`: tokenized data tsv, example: `./data/train_tokenized.tsv` which is generated by `./tools/pre_tokenized.sh`.

- `numerical`: each line contains numerical data (`token_ids`, `type_ids` and `pos_ids`, `role_ids` for optional) , example: `./data/train.numerical.tsv` which is generated by `./tools/pre_numericalize.sh`.

It also supports the file with `.gz` suffix which is compressed by `gzip` command.

Choices of `file_format`:

- `file`: a file only.

- `filelist`: contains multiple files, each line is a file, example: `./data/train_filelist`.

#### training
This section defines training related settings:

`init_params`: initialized parameters.

`batch_size`, `lr`, `num_epochs` and so on.

`log_dir`: the output path of training logs, include the log file (`${log_dir}/workerlog.${DEV_ID}`) of each GPU trainer. If `log_dir=""`, then the output of all GPU trainers will output to standard output.

`save_path`: the output path of saved parameters.

You can define other arguments in training script, such as:

```
train_args="--max_src_len 384 --max_seq_len 512"
```

## Disclaimer
This project aims to facilitate further research progress in dialogue generation. Baidu is not responsible for the 3rd party's generation with the pre-trained system.

## Contact information
For help or issues using Knover, please submit a GitHub issue.
