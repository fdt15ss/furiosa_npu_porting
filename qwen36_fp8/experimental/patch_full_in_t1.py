import re,subprocess,time
s=open('kx/full_in_noctx_noar.tcl').read()
# insert tiled axis + rope after gather; tile x before each rotation; slice back after
s=s.replace('    rope_val: [Three, T, Dhr, Ry, Rx]/bf16 = gather(rope_table_logical, position_ids_logical, Re)\n',
 '    rope_val0: [Three, T, Dhr, Ry, Rx]/bf16 = gather(rope_table_logical, position_ids_logical, Re)\n    T2: Axis = 2\n    rope_val: [Three, T2, Dhr, Ry, Rx]/bf16 = concat([rope_val0, rope_val0])\n')
s=re.sub(r'rope_val_il: \[T, Dhr, Ry, Rx\]','rope_val_il: [T2, Dhr, Ry, Rx]',s)
s=s.replace('dpe.exec(%0,%1 -> [T, Dhr, Ry, Rx])\n        write.to([T, Dhr, Ry, Rx]/bf16)','dpe.exec(%0,%1 -> [T2, Dhr, Ry, Rx])\n        write.to([T2, Dhr, Ry, Rx]/bf16)')
for suf,H in (('','Hq'),('_1','Hkv')):
    o,i=f'{H}_OUTER',f'{H}_INNER'
    b=re.search(rf'    xr_rot{suf}: .*?\n    \)\n',s,re.S).group(0)
    n=f'''    xr_c{suf}t: [T2, {o}, {i}, Rx, Dhr]/bf16 = concat([xr_c{suf}, xr_c{suf}])
    xr_rot{suf}t: [T2, {o}, {i}, Ry, Dhr]/bf16 = tk.EinsumByDpe(
        %0 = read(xr_c{suf}t)
        %1 = read(rope_val_il)
        %2 = dpe.exec(%0,%1 -> [T2, {o}, {i}, Ry, Dhr])
        write.to([T2, {o}, {i}, Ry, Dhr]/bf16)
    )
    xr_rot{suf}: [T, {o}, {i}, Ry, Dhr]/bf16 = slice(xr_rot{suf}t, T2, 0)
'''
    s=s.replace(b,n)
open('kx/full_in_t2.tcl','w').write(s)
t=time.time(); r=subprocess.run(['/usr/local/bin/furiosa-tcc','--stdin','--target-npu','renegade-8pe','--output','kx/full_in_t2.edf','--no-compiler-cache'],input=s,text=True,capture_output=True,timeout=570)
e=r.stderr+r.stdout; open('kx/fi4.out','w').write(e); st=re.findall(r'\[(\d+)/13\]',e)
print('t2 rc',r.returncode,'stage',st[-1] if st else '?',round(time.time()-t),'s',[l[:220] for l in e.splitlines() if 'failed' in l or 'rror' in l][-2:])
