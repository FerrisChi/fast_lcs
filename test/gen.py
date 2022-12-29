import os
from pathlib import Path
import subprocess
import os

PATH = '/home/chijj/gpu/build'
GEN = '/home/chijj/gpu/build/gen'
LCSs = ['cpu_fastlcs', 'cu_fastlcs', 'no_fastlcs']
# OUT_PATH = '/home/chijj/gpu/example/test'
OUT_PATH = '/home/chijj/gpu/example/test'
COUNT = 5

NUM_SEQs = [2,5,8,11,14]
LENGTHs = [50,50,50,50,50]

# result test
for i in range(COUNT):
    num_seq = NUM_SEQs[i]
    length = LENGTHs[i]
    out_path = f'{OUT_PATH}/{num_seq}_{length}'
    # f=open(out_path, mode='w')
    # f.close()
    # cmd = f'{GEN} -o {out_path} -n {num_seq} -l {length}'
    # os.system(cmd)

    ans = {}
    for prog in LCSs:
        cmd = f"{PATH}/{prog} --input {out_path}"
        print(cmd)
        ret = subprocess.getoutput(cmd)
        ans[prog] = ret
        
        # print(ret)
        rets = ret.split('\n')
        for ret in rets:
            if ret[5:7]=='bg' and ret[8:12]=='fast':
                st = float(ret.split(' ')[-1])
            if ret[5:7]=='ed' and ret[8:12]=='fast':
                ed = float(ret.split(' ')[-1])
        tm = ed - st
        print(tm)
    
    
    # for prog, ret in ans.items():
    #     print(prog)
    #     print(ret)