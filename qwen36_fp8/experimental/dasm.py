import sys
sys.path.insert(0,'pylib')
from elftools.elf.elffile import ELFFile
from capstone import Cs,CS_ARCH_X86,CS_MODE_64
P='/root/miniconda3/envs/furiosa02/lib/python3.11/site-packages/furiosa/llm_native.cpython-311-x86_64-linux-gnu.so'
b=open(P,'rb').read(); e=ELFFile(open(P,'rb')); t=e.get_section_by_name('.text'); ta,to,ts=t['sh_addr'],t['sh_offset'],t['sh_size']
def seg(v):
    for s in e.iter_segments():
        if s['p_type']=='PT_LOAD' and s['p_vaddr']<=v<s['p_vaddr']+s['p_filesz']: return s['p_offset']+v-s['p_vaddr']
md=Cs(CS_ARCH_X86,CS_MODE_64)
a,n=int(sys.argv[1],16),int(sys.argv[2],16)
import re
for x in md.disasm(b[seg(a):seg(a)+n],a):
    o=x.op_str; c=''
    m=re.search(r'rip ([+-]) 0x([0-9a-f]+)',o)
    if m:
        v=x.address+x.size+(1 if m[1]=='+' else -1)*int(m[2],16); f=seg(v)
        if f:
            s=b[f:f+60]; s=s.split(b'\0')[0][:48]
            if len(s)>=5 and all(32<=k<127 for k in s[:5]): c=' ; '+s.decode('latin1')
    print(hex(x.address)[2:],x.mnemonic,o,c)
