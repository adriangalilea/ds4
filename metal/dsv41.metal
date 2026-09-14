// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

/* Decode-time V4.1 hyper-connection glue, one token row, HC=4.
 *
 * The released graph runs each half-layer's HC work as separate dispatches:
 * split/sinkhorn, the pre-weighted collapse of the four streams, a BF16
 * rounding pass, the weighted RMSNorm and another BF16 pass (and, after the
 * sublayer, the post/comb expand plus its BF16 pass).  DeepSeek's production
 * decode does the whole thing in one "Mega-mHC" kernel.  These two kernels
 * are that fusion for ds4's graph: every reduction keeps the standalone
 * kernel's thread mapping and accumulation order, and every rounding point
 * is the same dsv41_bf16, so the outputs are byte-identical (checked by
 * tests/test_deepseek41_metal --hc-fuse). */
struct ds4_metal_args_dsv41_hc {
    uint  n_embd;
    uint  sinkhorn_iters;
    float hc_eps;
    float norm_eps;
    uint  copy_pre;
};

static inline float4 dsv41_bf16x4(float4 v) {
    return float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
}

/* kernel_dsv4_hc_split_sinkhorn's HC == 4 body, verbatim, on one lane. */
static inline void dsv41_hc_split4(device const float *mix, device const float *scale,
                                   device const float *base, device float *out,
                                   uint sinkhorn_iters, float epsv) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    const float4 pre_z = *((device const float4 *)mix) * pre_scale + *((device const float4 *)base);
    *((device float4 *)out) = ds4_hc_sigmoid(pre_z) + epsv;

    const float4 post_z = *((device const float4 *)(mix + 4)) * post_scale + *((device const float4 *)(base + 4));
    *((device float4 *)(out + 4)) = ds4_hc_twice_sigmoid(post_z);

    float4 r0 = *((device const float4 *)(mix +  8)) * comb_scale + *((device const float4 *)(base +  8));
    float4 r1 = *((device const float4 *)(mix + 12)) * comb_scale + *((device const float4 *)(base + 12));
    float4 r2 = *((device const float4 *)(mix + 16)) * comb_scale + *((device const float4 *)(base + 16));
    float4 r3 = *((device const float4 *)(mix + 20)) * comb_scale + *((device const float4 *)(base + 20));

    const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
    const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
    const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
    const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

    r0 = exp(r0 - m0);
    r1 = exp(r1 - m1);
    r2 = exp(r2 - m2);
    r3 = exp(r3 - m3);

    r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
    r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
    r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
    r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

    float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
    r0 *= col_inv;
    r1 *= col_inv;
    r2 *= col_inv;
    r3 *= col_inv;

    for (uint iter = 1; iter < sinkhorn_iters; ++iter) {
        r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
        r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
        r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
        r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

        col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;
    }

    *((device float4 *)(out +  8)) = r0;
    *((device float4 *)(out + 12)) = r1;
    *((device float4 *)(out + 16)) = r2;
    *((device float4 *)(out + 20)) = r3;
}

/* split(mix) -> split_out; x = bf16(sum_h pre[h] * residual[h]); norm =
 * bf16(rmsnorm(x) * weight).  `pre` is the coefficient row the previous
 * sublayer produced (V4.1's single-pass mHC), not this split's.  One
 * threadgroup of ds4_gpu_rms_norm_threads(n_embd) threads: the collapse
 * keeps kernel_dsv4_hc_weighted_sum's per-stream order and the norm keeps
 * kernel_rms_norm_mul_f32_4's loop, simd tree and 1/sqrt scale. */
kernel void kernel_dsv41_hc_collapse_norm4(
        constant ds4_metal_args_dsv41_hc &args,
        device const float *mix,
        device const float *scale,
        device const float *base,
        device const float *pre,
        device const float4 *residual,
        device       float *split_out,
        device       float4 *x_out,
        device const float4 *norm_weight,
        device       float4 *norm_out,
        threadgroup  float *shared [[threadgroup(0)]],
        ushort tid   [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    const uint n4 = args.n_embd >> 2;
    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float *sums = shared + args.n_embd;

    if (tid == 0) dsv41_hc_split4(mix, scale, base, split_out, args.sinkhorn_iters, args.hc_eps);
    if (sgitg == 0) sums[tiisg] = 0.0f;

    const float4 p = *((device const float4 *)pre);
    device const float4 *x0 = residual;
    device const float4 *x1 = residual + n4;
    device const float4 *x2 = residual + 2u * n4;
    device const float4 *x3 = residual + 3u * n4;
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        float4 v = 0.0f;
        v += x0[i] * p.x;
        v += x1[i] * p.y;
        v += x2[i] * p.z;
        v += x3[i] * p.w;
        v = dsv41_bf16x4(v);
        row[i] = v;
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) sums[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(sums[tiisg]);

    const float mean = sumf / (float)args.n_embd;
    const float norm_scale = 1.0f / sqrt(mean + args.norm_eps);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row[i];
        x_out[i] = v;
        norm_out[i] = dsv41_bf16x4((v * norm_scale) * norm_weight[i]);
    }
}

/* out[h] = bf16(post[h] * block + sum_s comb[h, s] * residual[s]) for the
 * four streams, kernel_dsv4_hc_expand4's index arithmetic and accumulation
 * order; copy_pre also carries split[0..3] into `pre` for the next layer. */
kernel void kernel_dsv41_hc_expand4_bf16(
        constant ds4_metal_args_dsv41_hc &args,
        device const float *block_out,
        device const float *residual,
        device const float *split,
        device       float *out,
        device       float *pre_out,
        uint d [[thread_position_in_grid]]) {
    if (d >= args.n_embd) return;
    device const float *post = split + 4;
    device const float *comb = split + 8;

    const float block_v = block_out[d];
    const float r0 = residual[d];
    const float r1 = residual[d + args.n_embd];
    const float r2 = residual[d + 2u * args.n_embd];
    const float r3 = residual[d + 3u * args.n_embd];

    for (uint dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * post[dst_hc];
        acc += comb[dst_hc + 0u * 4u] * r0;
        acc += comb[dst_hc + 1u * 4u] * r1;
        acc += comb[dst_hc + 2u * 4u] * r2;
        acc += comb[dst_hc + 3u * 4u] * r3;
        out[d + dst_hc * args.n_embd] = dsv41_bf16(acc);
    }
    if (args.copy_pre && d < 4) pre_out[d] = split[d];
}
