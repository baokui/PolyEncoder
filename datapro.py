import json
with open('/search/odin/guobk/data/vpaSupData/Q-all-train-20210819-sim.txt','r') as f:
    S = f.read().strip().split('\n')
S = [json.loads(s) for s in S]
R = []
for s in S:
    R.append(['1',s['input']]+s['click'])
    for t in s['neg']:
        R.append(['0',s['input'],t])
R = ['\t'.join(s) for s in R]
with open('/search/odin/guobk/data/data_polyEncode/vpa/train.txt','w') as f:
    f.write('\n'.join(R))

with open('/search/odin/guobk/data/vpaSupData/Q-all-test-20210819.json','r') as f:
    S = json.load(f)
R = []
for s in S:
    R.append(['1',s['input']]+s['pos'])
    for t in s['neg']:
        R.append(['0',s['input'],t])
R = ['\t'.join(s) for s in R]
with open('/search/odin/guobk/data/data_polyEncode/vpa/test.txt','w') as f:
    f.write('\n'.join(R))
with open('/search/odin/guobk/data/data_polyEncode/vpa/valid.txt','w') as f:
    f.write('\n'.join(R))