import os
files = os.listdir('/search/odin/guobk/data/data_polyEncode/vpa/train/')
files = [os.path.join('/search/odin/guobk/data/data_polyEncode/vpa/train',file) for file in files]
for i in range(len(files)):
    print(i)
    R = []
    with open(files[i],'r') as f:
        s = f.read().strip().split('\n')
    tok = '1'
    for j in range(len(s)):
        if s[j][0]!=tok:
            continue
        t = s[j].split('\t')
        R.append('\t'.join([t[0]]+t[2:]+[t[1]]))
        tok = str(1-int(tok))
    if len(R)%2!=0:
        R = R[:-1]
    with open(files[i],'w') as f:
        f.write('\n'.join(R))




# cache_file="/search/odin/guobk/data/data_polyEncode/ubuntu_data/train0.txt_[join_str]maxlen30_maxhis4_samplecntNone.cache"
# trainfile="/search/odin/guobk/data/data_polyEncode/ubuntu_data/train0.txt"
# with open('/search/odin/guobk/data/data_polyEncode/ubuntu_data/train1.txt','r') as f:
#     s0 = f.read().strip().split('\n')
# R0 = []
# R1 = []
# for i in range(len(s0)):
#     with open(trainfile,'w') as f:
#         f.write('\n'.join(s0[:-(i+1)]))
#     train_dataset = SelectionDataset(os.path.join('/search/odin/guobk/data/data_polyEncode/ubuntu_data', 'train{}.txt'.format(0)),context_transform, response_transform, sample_cnt=None)
#     train_dataloader = DataLoader(train_dataset,batch_size=32,collate_fn=train_dataset.batchify_join_str,shuffle=True)
#     os.remove(cache_file)
#     try:
#         for step, batch in enumerate(train_dataloader, start=1):
#             print(step)
#         R0.append(s0[i])
#         break
#     except:
#         R1.append(s0[i])

cache_file=""
train_dataset = SelectionDataset(os.path.join('/search/odin/guobk/data/data_polyEncode/vpa/train', 'train-{}.txt'.format('0')),context_transform, response_transform, sample_cnt=None)
train_dataloader = DataLoader(train_dataset,batch_size=32,collate_fn=train_dataset.batchify_join_str,shuffle=True)
# os.remove(cache_file)
for step, batch in enumerate(train_dataloader, start=1):
    print(step)

with open('/search/odin/guobk/data/data_polyEncode/vpa/train.txt','r') as f:
    S = f.read().strip().split('\n')
S = [s.split('\t') for s in S]
R = []
i = 0
r = []
query = ''
doc_pos = ''
doc_neg = ''
while i<len(S):
    if i%10000==0:
        print(i,len(S),len(R))
    if S[i][0]=='1':
        r = []
        if query:
            while doc_neg and doc_pos:
                n = doc_neg.pop()
                p = doc_pos.pop()
                r.append(['1',query,p])
                r.append(['0',query,n])
            R.extend(r)
        query = S[i][1]
        doc_pos = S[i][2:]
        doc_neg = []
        i+=1
        continue
    while i<len(S) and S[i][0]=='0':
        doc_neg.append(S[i][2])
        i+=1

n = int(len(R)/10)+1
i = 0
idx = 0
while i<len(R):
    with open('/search/odin/guobk/data/data_polyEncode/vpa/train_new/train-{}.txt'.format(idx),'w') as f:
        r = ['\t'.join(s) for s in R[i:i+n]]
        f.write('\n'.join(r))
    i+=n
    idx+=1
    print(idx)