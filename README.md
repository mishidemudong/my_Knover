2020 CCF BDCI: 千言对话baseline
本教程介绍使用基于paddlepaddle的大规模对话预训练框架Knover，及其提供的预训练模型，在"千言：多技能对话"比赛数据集上训练和测试。

0. 配置环境
获取github代码：

git clone https://github.com/PaddlePaddle/Knover
cd Knover
git checkout luge-dialogue
链接基线模型需要的配置：

ln  -s ./luge-dialogue/config ./config
1. 准备数据
模型训练和预测之前，都需要准备好所需的数据。首先，将比赛官网获取的数据放到当前环境中，具体目录可自行选择。数据获取可通过千言大赛官网或百度大脑AI Studio获取。

由于：

"千言：多技能对话"比赛的数据集较多，每个数据集之间存在差异
"千言：多技能对话"的基线模型在数据处理上，与Knover默认的处理略有不同
所以，无论是模型训练还是预测，都需要使用以下命令将数据转化成Knover可接受的id化的数据。

注意：在运行命令之前，一定要确保脚本中的输入、输出和参数均符合要求。在使用该命令处理训练集、验证集、测试集1和测试集2时，要分别修改脚本中的输入文件和参数列表，保证每个数据集与对应参数配置正确，保证输出文件正确。由于数据规模较大，脚本运行时间较长(尤其是训练集)，需要耐心等待，或自行分批次处理。

python ./luge-dialogue/tools/convert_data_to_numerical.py ./config/spm.model
最后，将id化的训练集、验证集和测试集放到data目录下(也可以放在其它目录下)，并在模型训练和测试的时候，保证数据路径与config目录下配置文件中，对应数据配置路径一致(必须).

2. 下载模型
本基线提供"大规模数据预训练模型"和"千言比赛数据微调模型"两个模型。

大规模数据预训练模型：采用包含20M对话session/60M对话utterance的大规模中文对话数据训练得到
千言比赛数据微调模型：上述预训练模型，继续在"千言：多技能对话"比赛的对话数据上进行微调得到
"千言：多技能对话"比赛共7个子数据集，3种对话类型：闲聊对话、知识对话、推荐对话。基线模型的输入除了数据token之外，还有一个对话类型token，用于区分不同的对话类型。

模型输入

模型下载方式如下：

# 大规模数据预训练模型
wget "https://dialogue.bj.bcebos.com/luge/12L.pretrain.tar"
tar -xvf ./12L.pretrain.tar

# 千言比赛数据微调模型
wget "https://dialogue.bj.bcebos.com/luge/12L.finetune.tar"
tar -xvf 12L.finetune.tar
3. 模型训练
模型训练命令如下：

# 热启模型，配置默认路径为"12L"，可在配置文件中设置
ln -snf 12L.pretrain 12L

# 模型训练，一定要确保GPU环境和模型参数配置正确，具体见下文示例
./scripts/local/train.sh ./config/12L_train.conf
训练之前，检查启动脚本与配置文件，确保配置正确：

配置GPU，位置：./scripts/local/train.sh
# 单GPU卡训练，以使用0号GPU卡为例
export CUDA_VISIBLE_DEVICES=0

# 多GPU卡训练，以使用0,1号GPU卡为例
export CUDA_VISIBLE_DEVICES=0,1
配置模型训练参数，位置：./config/12L_train.conf
# task settings
model=UnifiedTransformer # 模型类型
task=DialogGeneration # 任务类型

vocab_path="./config/vocab.txt" # 词典路径
spm_model_file="./config/spm.model" # sentencepiece模型路径
train_file="data/train.txt" # 训练数据路径，前述数据处理脚本得到的id化数据
valid_file="data/valid.txt" # 验证数据路径，前述数据处理脚本得到的id化数据
data_format="numerical" # 数据类型，表示已经id化
file_format="file" # 表示输入的是单个文件
config_path="./config/12L.json" # 模型网络配置路径

# training settings
init_params="12L" # 热启模型路径，即下载模型的解压路径
in_tokens="true" # batch类型
batch_size=8192 # batch大小
lr=1e-5 # 学习率
warmup_steps=1000 # warmup策略
weight_decay=0.01 # decay策略
num_epochs=20 # 训练轮数

train_args="--max_src_len 384 --max_tgt_len 128 --max_seq_len 512" # 输入数据长度，与数据处理脚本策略保持一致

log_steps=100 # 每多少步打印一次训练日志，需要根据数据集大小自行设置
validation_steps=1000 # 每多少步进行一次验证集验证，需要根据数据集大小自行设置
save_steps=1000 # 每多少步保存一次模型，需要根据数据集大小自行设置

log_dir="./log" # 日志路径
save_path="./output" # 模型输出路径
4. 模型预测
模型预测命令如下：

# 预测模型，配置默认路径为"12L"，可在配置文件中设置
ln -snf 12L.pretrain 12L

# 模型预测，一定要确保GPU环境和模型参数配置正确，具体见下文示例
./scripts/local/infer.sh ./config/12L_infer.conf
预测之前，检查启动脚本与配置文件，确保配置正确：

配置GPU，位置：./scripts/local/infer.sh
# 单GPU卡训练，以使用0号GPU卡为例
export CUDA_VISIBLE_DEVICES=0

# 多GPU卡训练，以使用0,1号GPU卡为例
export CUDA_VISIBLE_DEVICES=0,1
配置模型训练参数，位置：./config/12L_infer.conf
# task settings
model=UnifiedTransformer # 模型类型
task=DialogGeneration # 任务类型

vocab_path="./config/vocab.txt" # 词典路径
spm_model_file="./config/spm.model" # sentencepiece模型路径
infer_file="data/test.txt" # 测试数据路径，前述数据处理脚本得到的id化数据
data_format="numerical" # 数据类型，表示已经id化
file_format="file" # 表示输入的是单个文件
config_path="./config/12L.json" # 模型网络配置路径

# training settings
init_params="12L" # 测试模型路径，即下载模型的解压路径
batch_size=4 # batch大小，以样本为单位

output_name="response"  # 输出内容，这里指只输出最终的预测句子

infer_args="--do_generation true --decoding_strategy topk_sampling --num_samples 20 --topk 5 --is_cn true" # 解码策略，topk_sampling策略，topk=5，解码20次，取得分最高的回复最为最终结果

log_dir="./log" # 日志路径
save_path="./output" # 预测输出路径
5. 模型评估
模型评估需要将预测结果提交到官网进行评测：

step1：使用数据处理脚本将需要评测的数据样本化和id化
step2：进行模型预测，获取结果
step3：将预测结果准备成官网要求的格式，提交评估
6. 其它
本教程提供了"大规模数据预训练模型"和"千言比赛数据微调模型"两个模型，可作为一个基础baseline，帮助参赛者快速跑通整个参赛流程。 参赛者可以针对赛题进行其他改进，例如修改数据预处理方法，修改网络结构，修改训练方式，修改预测结果的后处理等。

注意论文所提的几个模型的训练参数需要注意修改和添加。有unitransmer, plato, nsp。其他基本都能走通，小问题已经修改。
默认参数batchsize 8192下，其中uni模型显存用量12G以内，plato显存31G，nsp18G左右。训练时间0.5小时/epoch , 1小时， 0.5小时。
