
import pandas as pd
from collections import Counter, defaultdict  #自带计数器
import csv
import operator #内部函数
from options import args
from utils import build_vocab, word_embeddings, fasttext_embeddings, gensim_to_fasttext_embeddings, gensim_to_embeddings, \
    reformat, write_discharge_summaries, concat_data, split_data



Y = 'full'

notes_file = '%s/NOTEEVENTS.csv' % args.MIMIC_3_DIR            #args.方法的使用

# # step 1: process code-related files
dfproc = pd.read_csv('%s/PROCEDURES_ICD.csv' % args.MIMIC_3_DIR)
dfdiag = pd.read_csv('%s/DIAGNOSES_ICD.csv' % args.MIMIC_3_DIR)

dfdiag['absolute_code'] = dfdiag.apply(lambda row: str(reformat(str(row[4]), True)), axis=1)
dfproc['absolute_code'] = dfproc.apply(lambda row: str(reformat(str(row[4]), False)), axis=1)

dfcodes = pd.concat([dfdiag, dfproc])


dfcodes.to_csv('%s/ALL_CODES.csv' % args.MIMIC_3_DIR, index=False,
           columns=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'absolute_code'],
           header=['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE'])

df = pd.read_csv('%s/ALL_CODES.csv' % args.MIMIC_3_DIR, dtype={"ICD9_CODE": str})
print("unique ICD9 code: {}".format(len(df['ICD9_CODE'].unique())))

# step 2: process notes
min_sentence_len = 3
disch_full_file = write_discharge_summaries("%s/disch_full.csv" % args.MIMIC_3_DIR, min_sentence_len, '%s/NOTEEVENTS.csv' % (args.MIMIC_3_DIR))
## disch_full 该文件是对文本文件进行去停，小写，添加SEP/CLS分隔符等等操作。

df = pd.read_csv('%s/disch_full.csv' % args.MIMIC_3_DIR)

df = df.sort_values(['SUBJECT_ID', 'HADM_ID'])
##原理类似于SQL中的order by
# step 3: filter out the codes that not emerge in notes
hadm_ids = set(df['HADM_ID'])
with open('%s/ALL_CODES.csv' % args.MIMIC_3_DIR, 'r') as lf:
    with open('%s/ALL_CODES_filtered.csv' % args.MIMIC_3_DIR, 'w') as of:
        w = csv.writer(of)
        w.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ADMITTIME', 'DISCHTIME'])
        r = csv.reader(lf)
        #header
        next(r)
        for i,row in enumerate(r):
            hadm_id = int(row[2])
            #print(hadm_id)
            #break
            if hadm_id in hadm_ids:
                w.writerow(row[1:3] + [row[-1], '', ''])

dfl = pd.read_csv('%s/ALL_CODES_filtered.csv' % args.MIMIC_3_DIR, index_col=None)

dfl = dfl.sort_values(['SUBJECT_ID', 'HADM_ID'])
dfl.to_csv('%s/ALL_CODES_filtered.csv' % args.MIMIC_3_DIR, index=False)

sorted_file = '%s/disch_full.csv' % args.MIMIC_3_DIR
df.to_csv(sorted_file, index=False)

# step 4: link notes with their code
labeled = concat_data('%s/ALL_CODES_filtered.csv' % args.MIMIC_3_DIR, sorted_file, '%s/notes_labeled.csv' % args.MIMIC_3_DIR)

dfnl = pd.read_csv(labeled)

# step 5: statistic unique word, total word, HADM_ID number
types = set()
num_tok = 0
for row in dfnl.itertuples():
    for w in row[3].split():
        types.add(w)
        num_tok += 1

print("num types", len(types), "num tokens", num_tok)
print("HADM_ID: {}".format(len(dfnl['HADM_ID'].unique())))
print("SUBJECT_ID: {}".format(len(dfnl['SUBJECT_ID'].unique())))

# step 6: split data into train dev test
fname = '%s/notes_labeled.csv' % args.MIMIC_3_DIR
base_name = "%s/disch" % args.MIMIC_3_DIR #for output
tr, dv, te = split_data(fname, base_name, args.MIMIC_3_DIR)

vocab_min = 1   #### 原本是3
vname = '%s/vocab.csv' % args.MIMIC_3_DIR
build_vocab(vocab_min, tr, vname)

#step 7: sort data by its note length, add length to the last column
 # 按注释长度对数据进行排序，将长度添加到最后一列   看不太明白
# for splt in ['train', 'dev', 'test']:   #  函数可以在311行进行检查
#     print('q')
#     filename = '%s/disch_%s_split.csv' % (args.MIMIC_3_DIR, splt)
#     print(filename)
#     df = pd.read_csv(filename)
#     df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
#     df = df.sort_values(['length'])
#     df.to_csv('%s/%s_full.csv' % (args.MIMIC_3_DIR, splt), index=False)

#step 8: train word embeddings via word2vec and fasttext
# w2v_file = word_embeddings('full', '%s/disch_full.csv' % args.MIMIC_3_DIR, 100, 0, 5)  # 100, 0, 5
# gensim_to_embeddings('%s/processed_full.w2v' % args.MIMIC_3_DIR, '%s/vocab.csv' % args.MIMIC_3_DIR, Y)
#
# fasttext_file = fasttext_embeddings('full', '%s/disch_full.csv' % args.MIMIC_3_DIR, 100, 0, 5)
# gensim_to_fasttext_embeddings('%s/processed_full.fasttext' % args.MIMIC_3_DIR, '%s/vocab.csv' % args.MIMIC_3_DIR, Y)

# step 9: statistic the top 50 code
Y = 50

counts = Counter()
dfnl = pd.read_csv('%s/notes_labeled.csv' % args.MIMIC_3_DIR)
for row in dfnl.itertuples():  # 生成一个元组
    for label in str(row[4]).split(';'):
        counts[label] += 1

codes_50 = sorted(counts.items(), key=operator.itemgetter(1), reverse=True) # 以列表返回可遍历的(键, 值) 元组数组

codes_50 = [code[0] for code in codes_50[:Y]] ### 此处可以人为划分，甚至可以找出最频繁的10个。

with open('%s/TOP_%s_CODES.csv' % (args.MIMIC_3_DIR, str(Y)), 'w') as of:
    w = csv.writer(of)
    for code in codes_50:
        w.writerow([code])

# step 10: split data according to train_50_hadm_ids dev... and test...
for splt in ['train', 'dev', 'test']:

    hadm_ids = set()
    with open('%s/%s_50_hadm_ids.csv' % (args.MIMIC_3_DIR, splt), 'r') as f:
        for line in f:
            hadm_ids.add(line.rstrip())  #   删除 string 字符串末尾的指定字符 默认空格
    with open('%s/notes_labeled.csv' % args.MIMIC_3_DIR, 'r') as f:
        with open('%s/%s_%s.csv' % (args.MIMIC_3_DIR, splt, str(Y)), 'w') as of:
            r = csv.reader(f)
            w = csv.writer(of)
            #header
            w.writerow(next(r))
            i = 0
            for row in r:
                try:
                     hadm_id = row[1]
                except:
                     continue
                if hadm_id not in hadm_ids:
                    continue
                codes = set(str(row[3]).split(';'))
                filtered_codes = codes.intersection(set(codes_50))
                if len(filtered_codes) > 0:
                    w.writerow(row[:3] + [';'.join(filtered_codes)])
                    i += 1

# step 11: sort data by its note length, add length to the last column 为什么要统计长度？？？

# for splt in ['train', 'dev', 'test']:
#     filename = '%s/%s_%s.csv' % (args.MIMIC_3_DIR, splt, str(Y))
#     df = pd.read_csv(filename)
#     df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
#     df = df.sort_values(['length'])
#     df.to_csv('%s/%s_%s.csv' % (args.MIMIC_3_DIR, splt, str(Y)), index=False)