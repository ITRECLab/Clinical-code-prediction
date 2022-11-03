import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from elmo.elmo import Elmo
import json
from utils import build_pretrain_embedding, load_embeddings
from math import floor
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np
import csv
## 2021-6-29 
##在output_layer + constive
##### transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import constants

###############

train_labels = ['401.9','38.93','428.0','427.31','414.01','96.04','96.6','584.9','250.00','96.71','272.4','518.81','99.04','39.61','599.0','530.81','96.72','272.0','285.9','88.56','244.9','486','38.91','285.1','36.15','276.2','496','99.15','995.92'
,'V58.61','507.0','038.9','88.72','585.9','403.90','311','305.1','37.22','412','33.24','39.95','287.5','410.71','276.1','V45.81','424.0','45.13','V15.82','511.9','37.23']
CODE_DESC_VECTOR_PATH = './data/mimic3/code_desc_vectors.csv'
## 输出标签描述的向量表示
EMBED_FILE_PATH = './data/mimic3/processed_full.fasttext.embed'
def load_embedding_weights():
    W = []
    # PAD and UNK already in embed file

    with open(EMBED_FILE_PATH) as ef:
        for line in ef:
            line = line.rstrip().split()
            vec = np.array(line[1:]).astype(np.float)
            # vec = vec / float(np.linalg.norm(vec) + 1e-6)
            W.append(vec)
    # logging.info(f'Total token count (including PAD, UNK) of full preprocessed discharge summaries: {len(W)}') ##人为注释
    weights = torch.tensor(W, dtype=torch.float)
    return weights

embed_weights = load_embedding_weights()


def load_label_embedding(labels,pad_index): ### 不可以直接将填充部分删除，因为向量的维度无法对齐
    code_desc = []
    desc_dt = {}
    max_desc_len = 0
    with open(f'{CODE_DESC_VECTOR_PATH}', 'r') as fin:
        for line in fin:
            items = line.strip().split()
            try:
                code = items[0]
            except:
                continue
            if code in labels:
                desc_dt[code] = list(map(int, items[1:]))
                max_desc_len = max(max_desc_len, len(desc_dt[code]))
    for code in labels:
        pad_len = max_desc_len - len(desc_dt[code])

        code_desc.append(desc_dt[code] + [pad_index] * pad_len)


    code_desc = torch.tensor(code_desc, dtype=torch.long)
    return code_desc

def embed_label_desc(label_desc):
    label_embeds = nn.Embedding.from_pretrained(label_desc,freeze=True).transpose(1, 2) ## 问题报错
    label_embeds = torch.div(label_embeds.squeeze(2), torch.sum(self.label_desc_mask, dim=-1).unsqueeze(1))

    return label_embeds
label_desc = load_label_embedding(train_labels, 0)



# file_name = './data/mimic3/TOP_50_CODES.csv'
# f = open(file_name, 'r')
# csvreader = csv.reader(f)
# train_labels = list(csvreader)
# print(train_labels)


# self.register_buffer('label_desc', label_desc)
# self.register_buffer('label_desc_mask', (self.label_desc != self.pad_idx) * 1.0)

######
## 处理transformer的系数

embed_size = 100
max_len = 1500
num_layers = 2
num_heads = 4
forward_expansion = 4
dropout_rate = 0.2
output_size = 50

device = 'cuda:0'
pad_idx = 0


# 2021-7-7 在output_layer中添加标签处理方法，用来改善对比学习；直接添加不方便，考虑在output layer中使用regist_buffer进行注册。
##将标签注意力矩阵作为自注意力矩阵的增强数据

class Attention(nn.Module):
    def __init__(self, hidden_size, output_size, attn_expansion, dropout_rate):
        super(Attention, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size*attn_expansion) ## 经过expansion后hidden-siz由128变成256
        ## 前面是拼接，后面是输出
        self.tnh = nn.Tanh()
        # self.dropout = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size*attn_expansion, output_size) ## outsize=50
    def forward(self, hidden, attn_mask=None):
        # output_1: B x S x H -> B x S x attn_expansion*H
        output_1 = self.tnh(self.l1(hidden))
        # output_2: B x S x attn_expansion*H -> B x S x output_size(O)
        output_2 = self.l2(output_1) # 与1相同，两个三维矩阵
        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)
        # attn_weights: B x S x output_size(O) -> B x O x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)
        # weighted_output: (B x O x S) @ (B x S x H) -> B x O x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout_rate, max_len):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) ## arange整型不含max-leng个数；扩充维度
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) ## 此处计算的意义在哪里（都是正值）
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1) ## 对pe的操作不是很清楚
#         self.register_buffer('pe', pe)
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :].to(x.device)
#         return self.dropout(x)
# class Transformer(nn.Module):
#     def __init__(self, embed_weights, embed_size,  max_len, num_layers, num_heads, forward_expansion,
#                  output_size, dropout_rate, label_desc, device):
#         super(Transformer, self).__init__()
#         if embed_size % num_heads != 0:
#             raise ValueError(f"Embedding size {embed_size} needs to be divisible by number of heads {num_heads}")
#         self.embed_size = embed_size
#         self.device = device
#         self.pad_idx = pad_idx
#         self.output_size = output_size
#
#         self.register_buffer('label_desc', label_desc)## 此处的两句源代码注释掉torch.tensor
#         self.register_buffer('label_desc_mask', (self.label_desc != self.pad_idx)*1.0)
#         self.label_attn = LabelAttention(embed_size, embed_size, dropout_rate)
#
#         self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=True)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_len)
#         encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
#                                                  dim_feedforward=forward_expansion*embed_size, dropout=dropout_rate)
#         self.encoder = TransformerEncoder(encoder_layers, num_layers)
#         self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for code in range(output_size)])
#
# ################################
#     def embed_label_desc(self):
#
#         label_embeds = self.embedder(self.label_desc).transpose(1, 2).matmul(self.label_desc_mask.unsqueeze(2)) ## 问题报错
#         label_embeds = torch.div(label_embeds.squeeze(2), torch.sum(self.label_desc_mask, dim=-1).unsqueeze(1))
#
#         return label_embeds ## div方法将输入的每个张量除以一个常量，返回新张量
# ################################
#
#
#
#     def forward(self, inputs, targets=None):
#
#         attn_mask = (inputs != self.pad_idx).unsqueeze(2).to(self.device)
#
#
#         src_key_padding_mask = (inputs == self.pad_idx).to(self.device)  # N x S
#
#         embeds = self.pos_encoder(self.embedder(inputs) * math.sqrt(self.embed_size))  # N x S x E
#         embeds = self.dropout(embeds)
#         embeds = embeds.permute(1, 0, 2)  # S x N x E
#
#         encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # T x N x E
#         encoded_inputs = encoded_inputs.permute(1, 0, 2)  # N x T x E
#
#         pooled_outputs = encoded_inputs.mean(dim=1)
#         outputs = torch.zeros((pooled_outputs.size(0), self.output_size)).to(self.device)
#
#         label_embeds = self.embed_label_desc()
#         weighted_output2, attn_weight2 = self.label_attn(encoded_inputs, label_embeds, attn_mask)
#
#         for code, fc in enumerate(self.fcs):
#             outputs[:, code:code+1] = fc(pooled_outputs)
#
#         return outputs
# class LabelAttention(nn.Module):
#     def __init__(self, hidden_size, label_embed_size, dropout_rate):
#         super(LabelAttention, self).__init__()
#         self.l1 = nn.Linear(hidden_size, label_embed_size)
#         self.tnh = nn.Tanh()
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, hidden, label_embeds, attn_mask=None):
#         # output_1: B x S x H -> B x S x E
#         output_1 = self.tnh(self.l1(hidden))
#         output_1 = self.dropout(output_1)
#
#         # print(output_1)
#         # print('i am the output_1')
#
#         # output_2: (B x S x E) x (E x L) -> B x S x L
#         output_2 = torch.matmul(output_1, label_embeds.t()) ## 当输入有多维时，将多出的一维提出来用作batch,其余的做矩阵乘法（.t转置一下）
#
#         # print(label_embeds.t())
#         print('i am the label-embed')
#
#
#         # Masked fill to avoid softmaxing over padded words
#         if attn_mask is not None:
#             output_2 = output_2.masked_fill(attn_mask == 0, -1e9)
#
#         # attn_weights: B x S x L -> B x L x S
#         attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)
#
#         # weighted_output: (B x L x S) @ (B x S x H) -> B x L x H
#         weighted_output = attn_weights @ hidden
#         return weighted_output, attn_weights
# dicts对应两个索引 args对应解析器里面的参数
class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()
        self.gpu = args.gpu
        if args.embed_file: # 默认为嵌入的向量文件
            print("loading pretrained embeddings from {}".format(args.embed_file)) #载入的是每个词对应的向量 format方法使用后面的字符代替{}
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'],
                                                                                     True)
                W = torch.from_numpy(pretrain_word_embedding) # 数组改为张量 二者共享内存
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0) # 建立词向量层 规定词嵌入个数，维度，以及补齐；

            # print(W.clone)
            self.embed.weight.data = W.clone() # 复制一个tensor 所使用的方法

        else: ### 重新训练
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim
        self.use_elmo = args.use_elmo
        if self.use_elmo:
            self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, requires_grad=args.elmo_tune,
                             dropout=args.elmo_dropout, gamma=args.elmo_gamma)
            with open(args.elmo_options_file, 'r') as fin:
                _options = json.load(fin)
            self.feature_size += _options['lstm']['projection_dim'] * 2
        self.embed_drop = nn.Dropout(p=args.dropout)
        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps], # 100
                     3: [self.feature_size, 150, 100, args.num_filter_maps],  # 150 100
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]}  # 200 150 100#

    # def load_freq(self):
    #     data = pd.read_csv('../code_freq.csv', dtype={'LENGTH': int})
    #     mlb = MultiLabelBinarizer()
    #
    #     # print(data['LABELS']) ## data[label]是一个标签组
    #
    #     data['LABELS'] = data['LABELS'].apply(lambda x: str(x).split(';'))
    #     code_counts = list(data['LABELS'].str.len())
    #     avg_code_counts = sum(code_counts) / len(code_counts)
    #     logging.info(f'In {split} set, average code counts per discharge summary: {avg_code_counts}')
    #     mlb.fit(data['LABELS'])  # 设置种类
    #     temp = mlb.transform(data['LABELS'])  ## 返回元组或者稀疏矩阵
    #     if mlb.classes_[-1] == 'nan':
    #         mlb.classes_ = mlb.classes_[:-1]
    #     logging.info(f'Final number of labels/codes: {len(mlb.classes_)}')
    #
    #     for i, x in enumerate(mlb.classes_):
    #         data[x] = temp[:, i]
    #     data.drop(['LABELS', 'LENGTH'], axis=1, inplace=True)
    #
    #     if data_setting == FULL:
    #         data = data[:-1]
    #
    #     code_list = list(mlb.classes_)
    #     print(code_list)
    #     print('qqqqqqqqqqqqqqq')# 最常见50个标签的列表
    #     label_freq = list(data[code_list].sum(axis=0))
    #     # print(label_freq) 一个代表频数的列表
    def forward(self, x, target, text_inputs):
        # print(x)##输入说明：x为文本，维度2*1200；target为标签，维度2*50；text-inputs，维度2*1200*50

        features = [self.embed(x)]
        # print(features)
        # print('i am feature')
        if self.use_elmo:
            elmo_outputs = self.elmo(text_inputs)
            elmo_outputs = elmo_outputs['elmo_representations'][0]
            features.append(elmo_outputs)


        x = torch.cat(features, dim=2)
        # print(x)
        # print('i am the cat')
        x = self.embed_drop(x)
        # print(x)
        # print('i am the embed one')
        return x
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
         ### (1-1matched)= target

    def forward(self, output1, output2, target, size_average=True):
        # print(output1.size())
        # print(output2.size())
        # print(target.size())


        device = torch.device('cuda:0')
        target = target.to(device)
        distances = (output2 - output1).pow(2).sum(1).to(device)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        # print(losses)
        # print("i am lossses")
        return losses.mean() if size_average else losses.sum()
class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()
        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)

        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss() ## 默认的两个参数
        # self.loss_function = nn.BCEWithLogitsLoss() ##原损失函数
        self.weight1 = torch.nn.Linear(300,1)
        self.weight2 = torch.nn.Linear(300,1)
        self.output_layer = torch.nn.Linear(300,50)

        self.lstm = torch.nn.LSTM(300, hidden_size=300, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(300, 200)
        self.linear_second = torch.nn.Linear(200,50)
        # self.weight1 = torch.nn.Linear(8921,1)
        # self.weight2 = torch.nn.Linear(300,1)
        # self.output_layer = torch.nn.Linear(300,8921)
        self.contrastive_loss = ContrastiveLoss(1.0)

        self.load_label_embedding = load_label_embedding(train_labels, 0)

    def forward(self, x, target, text_inputs):

##  2021-6-27尝试添加标签描述部分，具体是以建簇还是标签概率树还是其他方式待定
##  具体是对alpha进行操作(自适应聚类（动态K值）和对比学习（对比损失），两个损失来处理标签部分)
##  2021-7-1 加入对比效果不是很好，有两个点；第一，是否要给对比部分损失添加系数；借鉴动态K值的办法；第二，所对比的两部分是否可以再引入标签描述注意力；
##  第三，负样本的构造是否合理；


        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x)##注意力矩阵 2*50*300

        selfatt = torch.tanh(self.linear_first(x))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt,dim=1)
        selfatt = selfatt.transpose(1,2)
        self_att = torch.bmm(selfatt,x) # size= 2*50*300 文档本身使用自注意力不用动


        # label_desc = self.load_label_embedding(train_labels, 0)
        #
        # final_loss = self.contrastive_loss(label_desc, self_att, torch.stack([torch.tensor(1)]*150 + [torch.tensor(0)]*150))
        # print(final_loss,'i am final')


        weight1 = torch.sigmoid(self.weight1(m))

        weight2 = torch.sigmoid(self.weight2(self_att))

        weight1 = weight1/(weight1+weight2)
        weight2 = 1- weight1
        doc = weight1*m + weight2*self_att

        y = self.final.weight.mul(doc).sum(dim=2).add(self.final.bias)



        loss = self.loss_function(y, target)
        # loss = loss + final_loss
        # loss = self.loss_function(y, target)
        return y, loss
class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)
        # global filter_size
        # if args.filter_size.find(',') == -1:
        #     self.filter_num = 1
        filter_size = int(args.filter_size)
         # 添加了float类型


        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2 ))) # int(floor(filter_size / 2))
        xavier_uniform(self.conv.weight)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)



    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(float(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(float(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class ResCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True,
                                args.dropout)
            self.conv.add_module('conv-{}'.format(idx), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)

    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


    # def __init__(self, args, Y, dicts, label_freq=None, C=3.0, pad_idx=0):
    #     super(ResCNN, self).__init__()
    #
    #     self.word_rep = WordRep(args, Y, dicts)
    #
    #     self.conv = nn.ModuleList()
    #
    #     # if label_freq is not None:
    #     #     class_margin = torch.tensor(label_freq, dtype=torch.float32) ** 0.25  ###在这里相当于开了四次方
    #     #     class_margin = class_margin.masked_fill(class_margin == 0, 1)
    #     #     self.register_buffer('class_margin', 1.0 / class_margin)  ## 定义内存中的常量参数，前者为名字，后者为值（取倒数）
    #     #     self.C = C
    #     # else:
    #     #     self.class_margin = None
    #     #     self.C = 0
    #
    #
    #
    #     conv_dimension = self.word_rep.conv_dict[args.conv_layer]
    #     filter_sizes = args.filter_size.split(',')
    #     for idx in range(args.conv_layer):
    #         tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.dropout)
    #         self.conv.add_module('conv-{}'.format(idx), tmp)
    #
    #
    #     url = "E:/论文代码/Multi-Filter-Residual-Convolutional-Neural-Network-master/data/mimic3/code_freq.csv"
    #     data = pd.read_csv(url, dtype={'LENGTH': int})
    #     mlb = MultiLabelBinarizer()
    #
    #     # print(data['LABELS']) ## data[label]是一个标签组
    #
    #     # data['LABELS'] = data['LABELS'].apply(lambda x: str(x).split(';'))
    #     # code_counts = list(data['LABELS'].str.len())
    #     # avg_code_counts = sum(code_counts) / len(code_counts)
    #     # logging.info(f'In {split} set, average code counts per discharge summary: {avg_code_counts}')
    #     # mlb.fit(data['LABELS'])  # 设置种类
    #     # temp = mlb.transform(data['LABELS'])  ## 返回元组或者稀疏矩阵
    #     # if mlb.classes_[-1] == 'nan':
    #     #     mlb.classes_ = mlb.classes_[:-1]
    #     # logging.info(f'Final number of labels/codes: {len(mlb.classes_)}')
    #
    #     # for i, x in enumerate(mlb.classes_):
    #     #     data[x] = temp[:, i]
    #     # data.drop(['LABELS', 'LENGTH'], axis=1, inplace=True)
    #
    #
    # def forward(self, x, target, text_inputs):
    #
    #
    #
    #
    #
    #     code_list = ['038.9', '244.9', '250.00', '272.0', '272.4', '276.1', '276.2', '285.1', '285.9', '287.5', '305.1', '311', '33.24', '36.15', '37.22', '38.91', '38.93', '39.61', '39.95', '401.9', '403.90', '410.71', '412', '414.01', '424.0', '427.31', '428.0', '45.13', '486', '496', '507.0', '511.9', '518.81', '530.81', '584.9', '585.9', '599.0', '88.56', '88.72', '93.90', '96.04', '96.6', '96.71', '96.72', '99.04', '99.15', '995.92', 'V15.82', 'V45.81', 'V58.61']
    #     # label_freq = list(data[code_list].sum(axis=0))
    #     label_freq = [567, 761, 1416, 926, 1259, 494, 694, 726, 852, 471, 504, 493, 407, 719, 482, 773, 2139, 1096, 549, 3233, 513, 520, 477, 1921, 451, 1992, 2115, 415, 765, 646, 569, 421, 1186, 953, 1448, 544, 1067, 801, 530, 425, 1581, 1525, 1395, 969, 1287, 736, 613, 397, 479, 604]
    #     # print(label_freq) 一个代表频数的列表
    #
    #
    #
    #
    #
    #     x = self.word_rep(x, target, text_inputs)
    #
    #     x = x.transpose(1, 2)
    #
    #     for conv in self.conv:
    #         x = conv(x)
    #     x = x.transpose(1, 2)
    #
    #
    #
    #     class_margin = torch.tensor(label_freq, dtype=torch.float32) ** 0.25  ###在这里相当于开了四次方
    #     class_margin = class_margin.masked_fill(class_margin == 0, 1)
    #
    #     print(class_margin.size())
    #     print(target.size()) ##还是在计算target和margen上有问题，他俩尺寸和不同，margen尺寸（[50]）,target([2,50]),text_input有三通道。。。
    #     print(text_inputs.size())
    #     ldam_outputs =  text_inputs - target * class_margin * self.C ###outputs - targets * self.class_margin * self.C
    #
    #     self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps, label_freq, C=3.0)
    #
    #
    #
    #     y, loss = self.output_layer(x, target, text_inputs)
    #
    #     return y, loss
    #
    # def freeze_net(self):
    #     for p in self.word_rep.embed.parameters():
    #         p.requires_grad = False

class JLAN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')
        self.filter_num = len(filter_sizes)

        # self.transformer = Transformer(embed_weights, embed_size,max_len, num_layers, num_heads, forward_expansion,
        #          output_size, dropout_rate, label_desc, device)


        self.lstm = torch.nn.LSTM(300, hidden_size=300, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(300, 200)
        self.linear_second = torch.nn.Linear(200,50) ##### 此处注意，不同数据集需要更改

        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)
        #####
        self.embed_size = embed_size
        self.device = device
        self.pad_idx = pad_idx
        self.output_size = output_size

        self.embedder = nn.Embedding.from_pretrained(embed_weights, freeze=True)
        self.dropout = nn.Dropout(dropout_rate)
        # self.pos_encoder = PositionalEncoding(embed_size, dropout_rate, max_len)
        # encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
        #                                          dim_feedforward=forward_expansion * embed_size, dropout=dropout_rate)
        # self.encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fcs = nn.ModuleList([nn.Linear(embed_size, 1) for code in range(output_size)])



    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        # torch.Size([8, 2254, 150])

        # x = Trans(x)



        x = x.transpose(1, 2)

        # x = self.transformer(embed_weights, embed_size,  max_len, num_layers, num_heads, forward_expansion,
        #          output_size, dropout_rate, device)
        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)


        # selfatt = torch.tanh(self.linear_first(x))
        # selfatt = self.linear_second(selfatt)
        # selfatt = F.softmax(selfatt,dim=1)
        # selfatt = selfatt.transpose(1,2)
        # self_att = torch.bmm(selfatt,x)
        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False




import os
from pytorch_pretrained_bert.modeling import BertLayerNorm
from pytorch_pretrained_bert import BertModel, BertConfig
class Bert_seq_cls(nn.Module):

    def __init__(self, args, Y):
        super(Bert_seq_cls, self).__init__()

        print("loading pretrained bert from {}".format(args.bert_dir))

        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        self.config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(self.config))
        self.bert = BertModel.from_pretrained(args.bert_dir)

        self.dim_reduction = nn.Linear(self.config.hidden_size, args.num_filter_maps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.num_filter_maps, Y)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        x = self.dim_reduction(pooled_output)
        x = self.dropout(x)
        y = self.classifier(x)

        loss = F.binary_cross_entropy_with_logits(y, target)
        return y, loss

    def init_bert_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def freeze_net(self):
        pass


def pick_model(args, dicts):
    Y = len(dicts['ind2c'])
    if args.model == 'CNN':
        model = CNN(args, Y, dicts)
    elif args.model == 'MultiCNN':
        model = MultiCNN(args, Y, dicts)
    elif args.model == 'ResCNN':
        model = ResCNN(args, Y, dicts)
    elif args.model == 'DenCNN':
        model = DenCNN(args,Y,dicts)
    elif args.model == 'JLAN':
        model = MultiResCNN(args, Y, dicts)
    elif args.model == 'bert_seq_cls':
        model = Bert_seq_cls(args, Y)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model
