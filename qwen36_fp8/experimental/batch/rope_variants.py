"""Rewrite the rope_val_il einsum of a full_in DSL. usage: rope_variants.py <in.tcl> <split|pad> <out.tcl> [chunk]
split: compute rope_val_il in T chunks of <chunk> (default 4) and concat; pad: compute at 2T and slice."""
import sys
src, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 4
s = open(src).read()
old = '''    rope_val_il: [T, Dhr, Ry, Rx]/bf16 = tk.EinsumByDpe(
        %0 = read(rope_val)
        %1 = read(interleave_matrix)
        %2 = dpe.exec(%0,%1 -> [T, Dhr, Ry, Rx])
        write.to([T, Dhr, Ry, Rx]/bf16)
    )
'''
assert old in s
T = int(s.split(' T=')[1].split(',')[0])
def ein(name, src_t, ax):
    return f'''    {name}: [{ax}, Dhr, Ry, Rx]/bf16 = tk.EinsumByDpe(
        %0 = read({src_t})
        %1 = read(interleave_matrix)
        %2 = dpe.exec(%0,%1 -> [{ax}, Dhr, Ry, Rx])
        write.to([{ax}, Dhr, Ry, Rx]/bf16)
    )
'''
if mode == 'split':
    new = f'    TC: Axis = {chunk}\n'
    parts = []
    for j in range(T // chunk):
        new += f'    rope_val_c{j}: [Three, TC, Dhr, Ry, Rx]/bf16 = slice(rope_val, T, {j * chunk})\n'
        new += ein(f'rope_val_il_c{j}', f'rope_val_c{j}', 'TC')
        parts.append(f'rope_val_il_c{j}')
    new += f'    rope_val_il: [T, Dhr, Ry, Rx]/bf16 = concat([{", ".join(parts)}])\n'
elif mode == 'pad':
    new = f'    TP: Axis = {2 * T}\n    rope_val_p: [Three, TP, Dhr, Ry, Rx]/bf16 = concat([rope_val, rope_val])\n'
    new += ein('rope_val_il_p', 'rope_val_p', 'TP')
    new += '    rope_val_il: [T, Dhr, Ry, Rx]/bf16 = slice(rope_val_il_p, TP, 0)\n'
s = s.replace(old, new)
open(out, 'w').write(s)
print(out, T, mode)
