@compiler_config({
  "enable_einsum_fusion": true,
  "padding_policy": {
    "Small": 1.1
  },
  "instruction_mem_budget": 720896,
  "reshape_einsum_mode": {
    "Reshape": {
      "permute": true
    }
  },
  "tensor_unit_bridge_threshold_in_page": 12,
  "allow_external_operators": false,
  "partial_sum_policy": "Bf16ForSplitAndChipAndCluster",
  "enable_tactic_pruning": false,
  "scheduler_beam_search": true,
  "allow_reduce_by_ve_cluster_chip_reduce": false,
  "allow_reduce_by_ve_cluster_chip_reduce_base_population": false,
  "tactic_hint": {
    "ForLlmModelIOBound": 200
  },
  "num_transaction_simulation_per_pe": 1024,
  "dma_preference": 0.8,
  "propagate_sparse_axis_from_op": "None"
})
def full_in_broadcast<Bs=1, D=5120, DGroup=40, Dh=256, Dhr=32, Dq_INNER=6144, Dq_ORIGINAL=6144, Dq_OUTER=1, Dq_PADDED=6144, FusedQKVGroup_INNER=112, FusedQKVGroup_ORIGINAL=112, FusedQKVGroup_OUTER=1, FusedQKVGroup_PADDED=112, FusedQKV_INNER=14336, FusedQKV_ORIGINAL=14336, FusedQKV_OUTER=1, FusedQKV_PADDED=14336, Hkv_INNER=4, Hkv_ORIGINAL=4, Hkv_OUTER=1, Hkv_PADDED=4, Hq_INNER=24, Hq_ORIGINAL=24, Hq_OUTER=1, Hq_PADDED=24, Nb=8192, One=1, Re=262144, Rx=2, Ry=2, T=8, Three=3, eps=1e-06, mrope_section_height=11, mrope_section_time=11, mrope_section_width=10>(
    hidden_states: dram([Broadcast] | [T, D])/bf16 @ original=[T, D],
    norm_weight: dram([Broadcast] | [D])/bf16 @ original=[D],
    qkv_weight: dram([FusedQKV_ORIGINAL @ 14336 / 14336] | [FusedQKV_ORIGINAL @ 14336 % 14336, D])/f8_e4 @ original=[FusedQKV_ORIGINAL, D],
    qkv_weight_scale: dram([FusedQKVGroup_ORIGINAL @ 112 / 112] | [FusedQKVGroup_ORIGINAL @ 112 % 112, DGroup])/bf16 @ original=[FusedQKVGroup_ORIGINAL, DGroup],
    q_norm_weight: dram([Broadcast] | [Dh])/bf16 @ original=[Dh],
    k_norm_weight: dram([Broadcast] | [Dh])/bf16 @ original=[Dh],
    rope_table: dram([Broadcast] | [Re, Dhr, Ry, Rx])/bf16 @ original=[Re, Dhr, Ry, Rx],
    position_ids: dram([Broadcast] | [Three, T])/i32 @ original=[Three, T],
    k_cache: dram([Hkv_ORIGINAL @ 4 / 4] | [Nb, Bs, Hkv_ORIGINAL @ 4 % 4, Dh])/bf16 @ original=[Nb, Bs, Hkv_ORIGINAL, Dh],
    v_cache: dram([Hkv_ORIGINAL @ 4 / 4] | [Nb, Bs, Hkv_ORIGINAL @ 4 % 4, Dh])/bf16 @ original=[Nb, Bs, Hkv_ORIGINAL, Dh],
    kv_scatter_index: dram([Broadcast] | [One, T])/i32 @ original=[One, T],
    FusedQKV_ORIGINAL: Axis,
    FusedQKV_PADDED: Axis,
    FusedQKV_OUTER: Axis,
    FusedQKV_INNER: Axis,
    FusedQKVGroup_ORIGINAL: Axis,
    FusedQKVGroup_PADDED: Axis,
    FusedQKVGroup_OUTER: Axis,
    FusedQKVGroup_INNER: Axis,
    Hq_ORIGINAL: Axis,
    Hq_PADDED: Axis,
    Hq_OUTER: Axis,
    Hq_INNER: Axis,
    Dq_ORIGINAL: Axis,
    Dq_PADDED: Axis,
    Dq_OUTER: Axis,
    Dq_INNER: Axis,
    Hkv_ORIGINAL: Axis,
    Hkv_PADDED: Axis,
    Hkv_OUTER: Axis,
    Hkv_INNER: Axis
) -> dram([Dq_ORIGINAL @ 6144 / 6144] | [T, Dq_ORIGINAL @ 6144 % 6144])/bf16 @ original=[T, Dq_ORIGINAL], dram([Dq_ORIGINAL @ 6144 / 6144] | [T, Dq_ORIGINAL @ 6144 % 6144])/bf16 @ original=[T, Dq_ORIGINAL]:
    assert Bs == 1
    assert D == 5120
    assert DGroup == 40
    assert Dh == 256
    assert Dhr == 32
    assert Dq_INNER == 6144
    assert Dq_ORIGINAL == 6144
    assert Dq_OUTER == 1
    assert Dq_PADDED == 6144
    assert FusedQKVGroup_INNER == 112
    assert FusedQKVGroup_ORIGINAL == 112
    assert FusedQKVGroup_OUTER == 1
    assert FusedQKVGroup_PADDED == 112
    assert FusedQKV_INNER == 14336
    assert FusedQKV_ORIGINAL == 14336
    assert FusedQKV_OUTER == 1
    assert FusedQKV_PADDED == 14336
    assert Hkv_INNER == 4
    assert Hkv_ORIGINAL == 4
    assert Hkv_OUTER == 1
    assert Hkv_PADDED == 4
    assert Hq_INNER == 24
    assert Hq_ORIGINAL == 24
    assert Hq_OUTER == 1
    assert Hq_PADDED == 24
    assert Nb == 8192
    assert One == 1
    assert Re == 262144
    assert Rx == 2
    assert Ry == 2
    assert T == 8
    assert Three == 3
    Dh2: Axis = 512
    DhG: Axis = 256
    GroupSizeFusedQKV: Axis = 128
    GroupSizeD: Axis = 128
    FusedQKVGi: Axis = 112
    HqgiDh2: Axis = 12288
    HkviDh: Axis = 1024
    hidden_states_logical: [T, D]/bf16 = as_logical(hidden_states)
    norm_weight_logical: [D]/bf16 = as_logical(norm_weight)
    qkv_weight_logical_padded: [FusedQKV_PADDED, D]/f8_e4 = as_logical(qkv_weight)
    qkv_weight_logical: [FusedQKV_OUTER, FusedQKV_INNER, D]/f8_e4 = reshape(qkv_weight_logical_padded)
    qkv_weight_scale_logical_padded: [FusedQKVGroup_PADDED, DGroup]/bf16 = as_logical(qkv_weight_scale)
    qkv_weight_scale_logical: [FusedQKVGroup_OUTER, FusedQKVGroup_INNER, DGroup]/bf16 = reshape(qkv_weight_scale_logical_padded)
    q_norm_weight_logical: [Dh]/bf16 = as_logical(q_norm_weight)
    k_norm_weight_logical: [Dh]/bf16 = as_logical(k_norm_weight)
    rope_table_logical: [Re, Dhr, Ry, Rx]/bf16 = as_logical(rope_table)
    position_ids_logical: [Three, T]/i32 = as_logical(position_ids)
    k_cache_logical_padded: [Nb, Bs, Hkv_PADDED, Dh]/bf16 = as_logical(k_cache)
    k_cache_logical: [Nb, Bs, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = reshape(k_cache_logical_padded)
    v_cache_logical_padded: [Nb, Bs, Hkv_PADDED, Dh]/bf16 = as_logical(v_cache)
    v_cache_logical: [Nb, Bs, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = reshape(v_cache_logical_padded)
    kv_scatter_index_logical: [One, T]/i32 = as_logical(kv_scatter_index)
    x: [T, D]/bf16 = tk.ReduceByVe(
        %0 = read(hidden_states_logical)
        %1 = read(norm_weight_logical)
        %2 = ve.exec(%0 * %0)
        %3 = ve.exec(+ t[D] %2)
        %4 = ve.exec(%3 / f5120.0)
        %5 = ve.exec(%4 + f1e-06)
        %6 = ve.exec(sqrt %5)
        %7 = ve.exec(f1.0 / %6)
        %8 = ve.exec(%0 * %7)
        %9 = ve.exec(%8 * %1)
        %10 = ve.exec(%9 + %8)
        write.to([T, D]/bf16)
    )
    x_blk: [T, DGroup, GroupSizeD]/bf16 = reshape(x)
    x_scale: [T, DGroup]/f32 = tk.ReduceByVe(
        %0 = read(x_blk)
        %1 = ve.exec(abs_f %0)
        %2 = ve.exec(max_f t[GroupSizeD] %1)
        %3 = ve.exec(%2 / f448.0)
        %4 = ve.exec(%3 + f1e-12)
        write.to([T, DGroup]/f32)
    )
    x_q: [T, DGroup, GroupSizeD]/f8_e4 = tk.Elementwise(
        %0 = read(x_blk)
        %1 = read(x_scale)
        %2 = ve.exec(%0 / %1)
        write.to([T, DGroup, GroupSizeD]/f8_e4)
    )
    qkv_wb: [FusedQKV_OUTER, FusedQKVGi, GroupSizeFusedQKV, DGroup, GroupSizeD]/f8_e4 = reshape(qkv_weight_logical)
    qkv_s: [FusedQKV_OUTER, FusedQKVGi, DGroup]/bf16 = reshape(qkv_weight_scale_logical)
    qkv_blk: [T, FusedQKV_OUTER, FusedQKVGi, GroupSizeFusedQKV]/bf16 = tk.EinsumByDpe(
        %0 = read(qkv_wb)
        %1 = read(x_q)
        %2 = read(x_scale)
        %3 = read(qkv_s)
        %4 = dpe.exec(%0,%1 -> [T, FusedQKV_OUTER, FusedQKVGi, GroupSizeFusedQKV, DGroup])
        %5 = ve.exec(%4 * %2)
        %6 = ve.exec(%5 * %3)
        %7 = ve.exec(+ t[DGroup] %6)
        write.to([T, FusedQKV_OUTER, FusedQKVGi, GroupSizeFusedQKV]/bf16)
    )
    qkv_pc: [T, FusedQKV_OUTER, FusedQKV_INNER]/bf16 = reshape(qkv_blk)
    qg_i: [T, FusedQKV_OUTER, HqgiDh2]/bf16 = slice(qkv_pc, FusedQKV_INNER, 0)
    k_i: [T, FusedQKV_OUTER, HkviDh]/bf16 = slice(qkv_pc, FusedQKV_INNER, 12288)
    v_i: [T, FusedQKV_OUTER, HkviDh]/bf16 = slice(qkv_pc, FusedQKV_INNER, 13312)
    qg: [T, Hq_OUTER, Hq_INNER, Dh2]/bf16 = reshape(qg_i)
    q_raw: [T, Hq_OUTER, Hq_INNER, Dh]/bf16 = slice(qg, Dh2, 0)
    gate_hg: [T, Hq_OUTER, Hq_INNER, DhG]/bf16 = slice(qg, Dh2, 256)
    gate: [T, Hq_OUTER, Hq_INNER, Dh]/bf16 = reshape(gate_hg)
    k_raw: [T, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = reshape(k_i)
    v_h: [T, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = reshape(v_i)
    q_n: [T, Hq_OUTER, Hq_INNER, Dh]/bf16 = tk.ReduceByVe(
        %0 = read(q_raw)
        %1 = read(q_norm_weight_logical)
        %2 = ve.exec(%0 * %0)
        %3 = ve.exec(+ t[Dh] %2)
        %4 = ve.exec(%3 / f256.0)
        %5 = ve.exec(%4 + f1e-06)
        %6 = ve.exec(sqrt %5)
        %7 = ve.exec(f1.0 / %6)
        %8 = ve.exec(%0 * %7)
        %9 = ve.exec(%8 * %1)
        %10 = ve.exec(%9 + %8)
        write.to([T, Hq_OUTER, Hq_INNER, Dh]/bf16)
    )
    k_n: [T, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = tk.ReduceByVe(
        %0 = read(k_raw)
        %1 = read(k_norm_weight_logical)
        %2 = ve.exec(%0 * %0)
        %3 = ve.exec(+ t[Dh] %2)
        %4 = ve.exec(%3 / f256.0)
        %5 = ve.exec(%4 + f1e-06)
        %6 = ve.exec(sqrt %5)
        %7 = ve.exec(f1.0 / %6)
        %8 = ve.exec(%0 * %7)
        %9 = ve.exec(%8 * %1)
        %10 = ve.exec(%9 + %8)
        write.to([T, Hkv_OUTER, Hkv_INNER, Dh]/bf16)
    )
    RotA: Axis = 64
    Pass: Axis = 192
    rope_val: [Three, T, Dhr, Ry, Rx]/bf16 = gather(rope_table_logical, position_ids_logical, Re)
    interleave_matrix: [Three, Dhr]/bf16 = vector([1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0])
    rope_val_il: [T, Dhr, Ry, Rx]/bf16 = tk.EinsumByDpe(
        %0 = read(rope_val)
        %1 = read(interleave_matrix)
        %2 = dpe.exec(%0,%1 -> [T, Dhr, Ry, Rx])
        write.to([T, Dhr, Ry, Rx]/bf16)
    )
    xr: [T, Hq_OUTER, Hq_INNER, RotA]/bf16 = slice(q_n, Dh, 0)
    xp: [T, Hq_OUTER, Hq_INNER, Pass]/bf16 = slice(q_n, Dh, 64)
    xr_c: [T, Hq_OUTER, Hq_INNER, Rx, Dhr]/bf16 = reshape(xr)
    xr_rot: [T, Hq_OUTER, Hq_INNER, Ry, Dhr]/bf16 = tk.EinsumByDpe(
        %0 = read(xr_c)
        %1 = read(rope_val_il)
        %2 = dpe.exec(%0,%1 -> [T, Hq_OUTER, Hq_INNER, Ry, Dhr])
        write.to([T, Hq_OUTER, Hq_INNER, Ry, Dhr]/bf16)
    )
    x_rot: [T, Hq_OUTER, Hq_INNER, RotA]/bf16 = reshape(xr_rot)
    x_out: [T, Hq_OUTER, Hq_INNER, Dh]/bf16 = concat([x_rot, xp])
    xr_1: [T, Hkv_OUTER, Hkv_INNER, RotA]/bf16 = slice(k_n, Dh, 0)
    xp_1: [T, Hkv_OUTER, Hkv_INNER, Pass]/bf16 = slice(k_n, Dh, 64)
    xr_c_1: [T, Hkv_OUTER, Hkv_INNER, Rx, Dhr]/bf16 = reshape(xr_1)
    xr_rot_1: [T, Hkv_OUTER, Hkv_INNER, Ry, Dhr]/bf16 = tk.EinsumByDpe(
        %0 = read(xr_c_1)
        %1 = read(rope_val_il)
        %2 = dpe.exec(%0,%1 -> [T, Hkv_OUTER, Hkv_INNER, Ry, Dhr])
        write.to([T, Hkv_OUTER, Hkv_INNER, Ry, Dhr]/bf16)
    )
    x_rot_1: [T, Hkv_OUTER, Hkv_INNER, RotA]/bf16 = reshape(xr_rot_1)
    x_out_1: [T, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = concat([x_rot_1, xp_1])
    k_for_cache: [T, Bs, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = reshape(x_out_1)
    v_for_cache: [T, Bs, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = reshape(v_h)
    kv_idx_flat: [T]/i32 = reshape(kv_scatter_index_logical)
    updated_cache: [Nb, Bs, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = scatter(k_for_cache, kv_idx_flat, Nb, init=k_cache_logical)
    _t0: [Nb, Bs, Hkv_PADDED, Dh]/bf16 = reshape(updated_cache)
    _t1: dram([Hkv_ORIGINAL @ 4 / 4] | [Nb, Bs, Hkv_ORIGINAL @ 4 % 4, Dh])/bf16 @ original=[Nb, Bs, Hkv_ORIGINAL, Dh] = as_dram(_t0)
    _: dram([Hkv_ORIGINAL @ 4 / 4] | [Nb, Bs, Hkv_ORIGINAL @ 4 % 4, Dh])/bf16 @ original=[Nb, Bs, Hkv_ORIGINAL, Dh] = copy(src=_t1, dst=k_cache)
    updated_cache_1: [Nb, Bs, Hkv_OUTER, Hkv_INNER, Dh]/bf16 = scatter(v_for_cache, kv_idx_flat, Nb, init=v_cache_logical)
    _t2: [Nb, Bs, Hkv_PADDED, Dh]/bf16 = reshape(updated_cache_1)
    _t3: dram([Hkv_ORIGINAL @ 4 / 4] | [Nb, Bs, Hkv_ORIGINAL @ 4 % 4, Dh])/bf16 @ original=[Nb, Bs, Hkv_ORIGINAL, Dh] = as_dram(_t2)
    __1: dram([Hkv_ORIGINAL @ 4 / 4] | [Nb, Bs, Hkv_ORIGINAL @ 4 % 4, Dh])/bf16 @ original=[Nb, Bs, Hkv_ORIGINAL, Dh] = copy(src=_t3, dst=v_cache)
    q_out: [T, Dq_OUTER, Dq_INNER]/bf16 = reshape(x_out)
    gate_out: [T, Dq_OUTER, Dq_INNER]/bf16 = reshape(gate)
    _t4: [T, Dq_PADDED]/bf16 = reshape(q_out)
    q_out_1: dram([Dq_ORIGINAL @ 6144 / 6144] | [T, Dq_ORIGINAL @ 6144 % 6144])/bf16 @ original=[T, Dq_ORIGINAL] = as_dram(_t4)
    _t5: [T, Dq_PADDED]/bf16 = reshape(gate_out)
    gate_out_1: dram([Dq_ORIGINAL @ 6144 / 6144] | [T, Dq_ORIGINAL @ 6144 % 6144])/bf16 @ original=[T, Dq_ORIGINAL] = as_dram(_t5)
    return q_out_1, gate_out_1