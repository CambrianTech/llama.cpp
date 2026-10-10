#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];

#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // input state layout [S_v, S_v, H, n_seqs] (s0 only): per-seq stride is H*D.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float * s_ptr = (device const float *) (s) + state_in_base;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K, only slots 0..n_tokens-1 are written; older slots are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v * S_v * H * n_seqs
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = simd_sum(s_k);

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d;

            y += ls[j]*q_ptr[is];
        }

        y = simd_sum(y);

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float * dst_state = (device float *) (dst) + attn_size + (uint)target_slot * state_size_per_snap + state_out_base;
                FOR_UNROLL (short j = 0; j < NSG; j++) {
                    const short is = tx*NSG + j;
                    dst_state[is] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float * dst_state = (device float *) (dst) + attn_size + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

#else
// a simplified version of the above
// no performance improvement, so keep the above version for now

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B
    const uint i21 = tgpig.y; // H
    const uint i20 = tgpig.x*NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20;

    float lsf[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        lsf[j] = s_ptr[is*S_v];
    }

    thread T * ls = (thread T *) (lsf);

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr  = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr  = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        device const T * qt_ptr = (device const T *) (q_ptr);
        device const T * kt_ptr = (device const T *) (k_ptr);
        device const T * gt_ptr = (device const T *) (g_ptr);

        if (G == 1) {
            *ls *= exp(g_ptr[0]);
        } else {
            // KDA
            *ls *= exp(gt_ptr[tx]);
        }

        const float s_k = simd_sum(dot(*ls, kt_ptr[tx]));

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        *ls += kt_ptr[tx]*d;

        const float y = simd_sum(dot(*ls, qt_ptr[tx]));

        if (tx == 0) {
            *dst_attn = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        dst_attn += args.ne21*S_v;
    }

    device float * dst_state  = (device float *) (dst) + args.ne23*args.ne22*args.ne21*S_v + (i23*args.ne21 + i21)*S_v*S_v + i20;
    device T     * dstt_state = (device T     *) (dst_state);

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is*S_v] = lsf[j];
    }

#undef S_v
#undef G
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;
#endif


// GATED_DELTA_NET_BACK — the Metal twin of the CPU reference (ggml-cpu ops.cpp) and the CUDA
// kernel (ggml-cuda/gated_delta_net_back.cu), same math, same layout, same checkpointing.
// One threadgroup per (head, sequence), S_v threads; thread j owns row j of the state
// (kept transposed, M[j][i] = S[i][j]):
//   S' = a (.) S_prev (a = exp(g), scalar or per-row KDA),  u = S'^T k,  delta = beta (v - u),
//   S = S' + k delta^T,  o = scale S^T q
// The forward saves a checkpoint every C ~ sqrt(T) tokens; the backward walks the segments
// last to first, re-running each forward from its checkpoint, carrying dS. Row state lives in
// device scratch (after dst: 2 x S_v^2 floats is 128 KB at S_v = 128, above threadgroup
// memory); only per-token vectors and the column accumulators are in threadgroup memory.
// Output packed as [dq | dk | dv | dg | dbeta | dstate], each dense in its input's order;
// dq/dk accumulate across the heads that share them (broadcast), so they are zeroed first
// and added with device atomics. Column sums inside a threadgroup (dq, dk, the KDA dg) are a
// column pass: thread i walks column i of the rows the other threads wrote, after a device
// barrier (Metal has no threadgroup float atomics).

kernel void kernel_gated_delta_net_back_zero(
        device float * dst,
        constant uint64_t & n,
        uint tpig[[thread_position_in_grid]]) {
    if (tpig < n) {
        dst[tpig] = 0.0f;
    }
}

template<short S_v>
kernel void kernel_gated_delta_net_back_impl(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const float * q,
        device const float * k,
        device const float * v,
        device const float * g,
        device const float * beta,
        device const float * s0,
        device const float * grad,
        device       float * out,
        device       float * ckpt,
        device       float * seg,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int SS  = S_v*S_v;
    constexpr int NSG = S_v/32 > 0 ? S_v/32 : 1;

    threadgroup float a_s[S_v], q_s[S_v], k_s[S_v], v_s[S_v], gy_s[S_v];
    threadgroup float delta_s[S_v], u_s[S_v], dd_s[S_v], du_s[S_v];
    threadgroup float red[NSG];

    const int     j   = tiitg;
    const int64_t iv1 = tgpig.x; // head
    const int64_t iv3 = tgpig.y; // sequence

    const int64_t H = args.H, T = args.T, C = args.C;
    const int64_t rq3 = args.n_seqs/args.neq3;
    const int64_t rk3 = args.n_seqs/args.nek3;
    const bool    kda = args.kda != 0;
    const int64_t neg0 = kda ? S_v : 1;

    device float * dq = out;
    device float * dk = out + args.off_dk;
    device float * dv = out + args.off_dv;
    device float * dg = out + args.off_dg;
    device float * db = out + args.off_db;
    device float * ds = out + args.off_ds;

    device atomic_float * dq_at = (device atomic_float *) dq;
    device atomic_float * dk_at = (device atomic_float *) dk;

    const int64_t n_ck = (T + C - 1)/C;
    ckpt += (iv3*H + iv1)*n_ck*SS;
    seg  += (iv3*H + iv1)*(C + 3)*SS;
    device float * Mp = seg + (C + 1)*SS;
    device float * dM = Mp + SS;

    device const float * s0_blk = s0 + iv3*H*SS + iv1*SS;
    const int64_t snap = (int64_t) SS*H*args.n_seqs;
    device const float * gy_all = grad;
    device const float * gs_all = grad + (int64_t) S_v*H*T*args.n_seqs;

    // a sum over the threadgroup, valid in every thread
    auto tg_sum = [&](float x) {
        float r = simd_sum(x);
        if (NSG > 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tiisg == 0) {
                red[sgitg] = r;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            r = 0.0f;
            for (int s = 0; s < NSG; ++s) {
                r += red[s];
            }
        }
        return r;
    };

    auto q_at = [&](int64_t t) { return q + (iv3/rq3)*args.sq3 + t*args.sq2 + (iv1 % args.neq1)*args.sq1; };
    auto k_at = [&](int64_t t) { return k + (iv3/rk3)*args.sk3 + t*args.sk2 + (iv1 % args.nek1)*args.sk1; };
    auto v_at = [&](int64_t t) { return v + iv3*args.sv3 + t*args.sv2 + iv1*args.sv1; };
    auto g_at = [&](int64_t t) { return g + (iv3*T + t)*H*neg0 + iv1*neg0; };
    auto b_at = [&](int64_t t) { return beta[(iv3*T + t)*H + iv1]; };

    // one forward step of row j: Mnext_j = a (.) Mprev_j + k delta_j
    auto fwd_step = [&](device const float * Mprev, device float * Mnext, int64_t t) {
        device const float * g_t = g_at(t);
        a_s[j] = exp(kda ? g_t[j] : g_t[0]);
        k_s[j] = k_at(t)[j];
        v_s[j] = v_at(t)[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float b = b_at(t);
        float u = 0.0f;
        for (int i = 0; i < S_v; ++i) {
            u += Mprev[i]*a_s[i]*k_s[i];
        }
        const float delta = (v_s[j] - u)*b;
        for (int i = 0; i < S_v; ++i) {
            Mnext[i] = Mprev[i]*a_s[i] + k_s[i]*delta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    };

    // ---- forward, checkpointing every C tokens
    for (int i = 0; i < S_v; ++i) {
        seg[j*S_v + i] = s0_blk[j*S_v + i];
    }
    for (int64_t t = 0; t < T; ++t) {
        device float * cur = seg + (t & 1)*SS + j*S_v;
        device float * nxt = seg + ((t + 1) & 1)*SS + j*S_v;
        if (t % C == 0) {
            device float * ck = ckpt + (t/C)*SS + j*S_v;
            for (int i = 0; i < S_v; ++i) {
                ck[i] = cur[i];
            }
        }
        fwd_step(cur, nxt, t);
    }

    // ---- backward, carrying dM
    for (int i = 0; i < S_v; ++i) {
        dM[j*S_v + i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int64_t s = n_ck - 1; s >= 0; --s) {
        const int64_t t0 = s*C;
        const int64_t t1 = t0 + C < T ? t0 + C : T;
        {
            device const float * ck = ckpt + s*SS + j*S_v;
            for (int i = 0; i < S_v; ++i) {
                seg[j*S_v + i] = ck[i];
            }
            for (int64_t tt = t0; tt < t1; ++tt) {
                fwd_step(seg + (tt - t0)*SS + j*S_v, seg + (tt - t0 + 1)*SS + j*S_v, tt);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (int64_t t = t1 - 1; t >= t0; --t) {
            device const float * g_t = g_at(t);
            a_s[j]  = exp(kda ? g_t[j] : g_t[0]);
            q_s[j]  = q_at(t)[j];
            k_s[j]  = k_at(t)[j];
            v_s[j]  = v_at(t)[j];
            gy_s[j] = gy_all[((iv3*T + t)*H + iv1)*S_v + j];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float b = b_at(t);
            device float       * dMj = dM + j*S_v;
            device float       * Mpj = Mp + j*S_v;
            device const float * M1  = seg + (t - t0 + 1)*SS;      // S after token t (all rows)
            device const float * M0j = seg + (t - t0)*SS + j*S_v;  // S before token t, row j

            // row pass A: the snapshot gradient, the output's dS, and S', u, delta
            const int64_t slot = T - 1 - t;
            if (slot < args.K) {
                device const float * gs = gs_all + slot*snap + (iv3*H + iv1)*SS + j*S_v;
                for (int i = 0; i < S_v; ++i) {
                    dMj[i] += gs[i];
                }
            }
            {
                const float gyj = gy_s[j];
                for (int i = 0; i < S_v; ++i) {
                    dMj[i] += args.scale*gyj*q_s[i];
                }
            }
            float u = 0.0f;
            for (int i = 0; i < S_v; ++i) {
                const float mp = M0j[i]*a_s[i];
                Mpj[i] = mp;
                u += mp*k_s[i];
            }
            u_s[j]     = u;
            delta_s[j] = (v_s[j] - u)*b;

            // row pass B: dd_j = sum_i dM[j][i] k_i ; dv ; du
            {
                float dd = 0.0f;
                for (int i = 0; i < S_v; ++i) {
                    dd += dMj[i]*k_s[i];
                }
                dd_s[j] = dd;
                dv[((iv3*T + t)*H + iv1)*S_v + j] = b*dd;
                du_s[j] = -b*dd;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // column pass C (thread j = column j): dq_j = scale sum_r S[r][j] gy_r ;
            // dk_j = sum_r dM[r][j] delta_r + S'[r][j] du_r   (dM before the du update)
            {
                float cq = 0.0f, ck = 0.0f;
                for (int r = 0; r < S_v; ++r) {
                    cq += M1[r*S_v + j]*gy_s[r];
                    ck += dM[r*S_v + j]*delta_s[r] + Mp[r*S_v + j]*du_s[r];
                }
                atomic_fetch_add_explicit(&dq_at[(((iv3/rq3)*T + t)*args.neq1 + iv1 % args.neq1)*S_v + j], args.scale*cq, memory_order_relaxed);
                atomic_fetch_add_explicit(&dk_at[(((iv3/rk3)*T + t)*args.nek1 + iv1 % args.nek1)*S_v + j], ck, memory_order_relaxed);
            }
            {
                const float dbeta = tg_sum(dd_s[j]*(v_s[j] - u_s[j]));
                if (j == 0) {
                    db[(iv3*T + t)*H + iv1] = dbeta;
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // row pass D: u = S'^T k -> dM[j][i] += k_i du_j ; the scalar gate's row sum
            float dg_local = 0.0f;
            {
                const float duj = du_s[j];
                for (int i = 0; i < S_v; ++i) {
                    dMj[i] += k_s[i]*duj;
                    dg_local += dMj[i]*Mpj[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // S' = a (.) S_prev: dg = sum dS' (.) S' (column j for KDA, a scalar otherwise)
            if (kda) {
                float cg = 0.0f;
                for (int r = 0; r < S_v; ++r) {
                    cg += dM[r*S_v + j]*Mp[r*S_v + j];
                }
                dg[((iv3*T + t)*H + iv1)*S_v + j] = cg;
            } else {
                const float dg0 = tg_sum(dg_local);
                if (j == 0) {
                    dg[(iv3*T + t)*H + iv1] = dg0;
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            // row pass E: dS_prev = a (.) dS'
            for (int i = 0; i < S_v; ++i) {
                dMj[i] *= a_s[i];
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }

    for (int i = 0; i < S_v; ++i) {
        ds[(iv3*H + iv1)*SS + j*S_v + i] = dM[j*S_v + i];
    }
}

typedef decltype(kernel_gated_delta_net_back_impl<32>) kernel_gated_delta_net_back_t;

template [[host_name("kernel_gated_delta_net_back_f32_32")]]  kernel kernel_gated_delta_net_back_t kernel_gated_delta_net_back_impl<32>;
template [[host_name("kernel_gated_delta_net_back_f32_64")]]  kernel kernel_gated_delta_net_back_t kernel_gated_delta_net_back_impl<64>;
template [[host_name("kernel_gated_delta_net_back_f32_128")]] kernel kernel_gated_delta_net_back_t kernel_gated_delta_net_back_impl<128>;
