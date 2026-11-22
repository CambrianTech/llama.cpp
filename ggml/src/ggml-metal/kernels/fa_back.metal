#include "common.h"
#include "dequantize.h"

// FLASH_ATTN_BACK: the gradients of FLASH_ATTN_EXT, recomputing the scores instead of keeping
// them, so the memory is the gradients alone (the CPU reference: ggml-cpu ops.cpp
// ggml_compute_forward_flash_attn_back_f32). One simdgroup per row; each lane holds every 32nd
// element of a row. Pass Q, per query row and head: the row's log-sum-exp over the window, then
// in one more sweep D = sum_j p_j dp_j, A = sum_j p_j dp_j k_j and B = sum_j p_j k_j, so
// dq = scale (A - D B). Pass KV, per KV position and KV head: dk and dv summed over the query
// rows of every head in its group, from the lse and D the Q pass wrote.

#define FA_BACK_MAX_D 512

template <typename T>
static inline float fa_back_elem(device const char * row, int c);

template <> inline float fa_back_elem<float>(device const char * row, int c) { return ((device const float *) row)[c]; }
template <> inline float fa_back_elem<half> (device const char * row, int c) { return (float) ((device const half *) row)[c]; }
template <> inline float fa_back_elem<block_q8_0>(device const char * row, int c) {
    device const block_q8_0 * b = (device const block_q8_0 *) row + c/QK8_0;
    return (float) b->d * (float) b->qs[c%QK8_0];
}

static inline float fa_back_mask(constant ggml_metal_kargs_flash_attn_back & args, device const char * mask, int j, int i, int h) {
    if (!args.has_mask) {
        return 0.0f;
    }
    return (float) ((device const half *) (mask + i*args.nbm1 + (h % args.nem2)*args.nbm2))[j];
}

template <typename T>
kernel void kernel_flash_attn_back_q(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * d,
        device       char * dst,
        uint  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int i  = tgpig % args.N;
    const int h  = tgpig / args.N;
    const int hk = h / (args.H / args.Hkv);

    const int nq = args.D  / 32;
    const int nv = args.Dv / 32;
    float qr[FA_BACK_MAX_D/32], dr[FA_BACK_MAX_D/32], a[FA_BACK_MAX_D/32], b[FA_BACK_MAX_D/32];
    for (int c = 0; c < nq; ++c) {
        qr[c] = *(device const float *) (q + (tiisg + 32*c)*sizeof(float) + i*args.nbq1 + h*args.nbq2);
        a[c] = 0.0f;
        b[c] = 0.0f;
    }
    for (int c = 0; c < nv; ++c) {
        dr[c] = *(device const float *) (d + (tiisg + 32*c)*sizeof(float) + h*args.nbd1 + i*args.nbd2);
    }

    // the log-sum-exp, online
    float m = -INFINITY;
    float l = 0.0f;
    for (int j = 0; j < args.M; ++j) {
        const float mj = fa_back_mask(args, mask, j, i, h);
        if (mj == -INFINITY) {
            continue;
        }
        device const char * kr = k + j*args.nbk1 + hk*args.nbk2;
        float s = 0.0f;
        for (int c = 0; c < nq; ++c) {
            s += qr[c]*fa_back_elem<T>(kr, tiisg + 32*c);
        }
        s = simd_sum(s)*args.scale + mj;
        const float mn = max(m, s);
        l = l*exp(m - mn) + exp(s - mn);
        m = mn;
    }
    const float lse = m == -INFINITY ? -INFINITY : m + log(l);

    // D, A and B in one sweep
    float dd = 0.0f;
    if (lse != -INFINITY) {
        for (int j = 0; j < args.M; ++j) {
            const float mj = fa_back_mask(args, mask, j, i, h);
            if (mj == -INFINITY) {
                continue;
            }
            device const char * kr = k + j*args.nbk1 + hk*args.nbk2;
            device const char * vr = v + j*args.nbv1 + hk*args.nbv2;
            float s = 0.0f;
            for (int c = 0; c < nq; ++c) {
                s += qr[c]*fa_back_elem<T>(kr, tiisg + 32*c);
            }
            float dp = 0.0f;
            for (int c = 0; c < nv; ++c) {
                dp += dr[c]*fa_back_elem<T>(vr, tiisg + 32*c);
            }
            const float p = exp(simd_sum(s)*args.scale + mj - lse);
            dp = simd_sum(dp);
            dd += p*dp;
            for (int c = 0; c < nq; ++c) {
                const float kc = fa_back_elem<T>(kr, tiisg + 32*c);
                a[c] += p*dp*kc;
                b[c] += p*kc;
            }
        }
    }

    device float * dq = (device float *) dst + (h*args.N + i)*args.D;
    for (int c = 0; c < nq; ++c) {
        dq[tiisg + 32*c] = args.scale*(a[c] - dd*b[c]);
    }
    if (tiisg == 0) {
        device float * lsev = (device float *) (dst + args.offs_lse);
        lsev[h*args.N + i] = lse;
        lsev[args.N*args.H + h*args.N + i] = dd;
    }
}

template <typename T>
kernel void kernel_flash_attn_back_kv(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * d,
        device       char * dst,
        uint  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int L  = args.M - args.kv0;
    const int j  = args.kv0 + tgpig % L;
    const int hk = tgpig / L;
    const int G  = args.H / args.Hkv;

    const int nq = args.D  / 32;
    const int nv = args.Dv / 32;
    float kr[FA_BACK_MAX_D/32], vr[FA_BACK_MAX_D/32], gk[FA_BACK_MAX_D/32], gv[FA_BACK_MAX_D/32];
    device const char * kp = k + j*args.nbk1 + hk*args.nbk2;
    device const char * vp = v + j*args.nbv1 + hk*args.nbv2;
    for (int c = 0; c < nq; ++c) {
        kr[c] = fa_back_elem<T>(kp, tiisg + 32*c);
        gk[c] = 0.0f;
    }
    for (int c = 0; c < nv; ++c) {
        vr[c] = fa_back_elem<T>(vp, tiisg + 32*c);
        gv[c] = 0.0f;
    }
    device const float * lsev = (device const float *) (dst + args.offs_lse);
    for (int h = hk*G; h < (hk + 1)*G; ++h) {
        for (int i = 0; i < args.N; ++i) {
            const float lse = lsev[h*args.N + i];
            const float mj  = fa_back_mask(args, mask, j, i, h);
            if (lse == -INFINITY || mj == -INFINITY) {
                continue;
            }
            device const char * qp = q + i*args.nbq1 + h*args.nbq2;
            device const char * dp = d + h*args.nbd1 + i*args.nbd2;
            float s = 0.0f;
            float t = 0.0f;
            for (int c = 0; c < nq; ++c) {
                s += ((device const float *) qp)[tiisg + 32*c]*kr[c];
            }
            for (int c = 0; c < nv; ++c) {
                t += ((device const float *) dp)[tiisg + 32*c]*vr[c];
            }
            const float p  = exp(simd_sum(s)*args.scale + mj - lse);
            const float ds = p*(simd_sum(t) - lsev[args.N*args.H + h*args.N + i])*args.scale;
            for (int c = 0; c < nq; ++c) {
                gk[c] += ds*((device const float *) qp)[tiisg + 32*c];
            }
            for (int c = 0; c < nv; ++c) {
                gv[c] += p*((device const float *) dp)[tiisg + 32*c];
            }
        }
    }
    device float * dk = (device float *) (dst + args.offs_dk) + (hk*L + j - args.kv0)*args.D;
    device float * dv = (device float *) (dst + args.offs_dv) + (hk*L + j - args.kv0)*args.Dv;
    for (int c = 0; c < nq; ++c) {
        dk[tiisg + 32*c] = gk[c];
    }
    for (int c = 0; c < nv; ++c) {
        dv[tiisg + 32*c] = gv[c];
    }
}


// THE TILED BACKWARD (FlashAttention-2 style, simdgroup matrices; D == Dv, D % 32 == 0). Query
// blocks of FAB_BR rows and KV blocks of FAB_BC positions; the four simdgroups of a threadgroup
// each compute one 8x8 tile of S = Q K^T or dP = dO V^T per block (inputs staged as half,
// accumulated in F32), and the gradients accumulate in simdgroup matrices split by column.
//   pass A (query blocks): each row's log-sum-exp and D = sum_j p_j dp_j, online in one sweep
//   pass B (KV blocks):    dK += dS^T Q, dV += P^T dO over every query row of the KV head's group
//   pass C (query blocks): dQ += dS K
#define FAB_BR  8
#define FAB_BC  16
#define FAB_NSG 4
// dO (and so dP, D, dS and the gradients) is carried through the half tiles scaled by the power of
// two that brings its largest magnitude to [4, 8) (ggml_flash_attn_back_scale; pass "scale" below):
// training gradients sit far below half's normal range (6.1e-5) and lost their precision there
// (test-walk-exact, M5 2026-10-14: 0.9943 against the F32 gradient unscaled, 1.0000 scaled)

#define FAB_SCALE_NT 1024 // the scale pass's threads (ggml_metal_op_flash_attn_back dispatches this many)

kernel void kernel_flash_attn_back_scale(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * d,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const ushort ntg = FAB_SCALE_NT;
    threadgroup float * smax = (threadgroup float *) shmem;
    float mx = 0.0f;
    for (int r = 0; r < args.N*args.H; ++r) {
        device const float * row = (device const float *) (d + (r % args.H)*args.nbd1 + (r / args.H)*args.nbd2);
        for (int c = tiitg; c < args.Dv; c += ntg) {
            mx = max(mx, fabs(row[c]));
        }
    }
    mx = simd_max(mx);
    if (tiisg == 0) {
        smax[sgitg] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg == 0) {
        for (int s = 1; s < ntg/32; ++s) {
            mx = max(mx, smax[s]);
        }
        float scale = 1.0f;
        if (mx > 0.0f) {
            int e;
            frexp(mx, e);
            scale = ldexp(1.0f, clamp(3 - e, -24, 40));
        }
        ((device float *) (dst + args.offs_lse))[2*args.N*args.H] = scale;
    }
}

// rows [r0, r0 + FAB_BR|FAB_BC) of a head into threadgroup half, zeros past `n` rows
// (rows outer, threads across a row's columns: no division per element)
template <typename T>
static inline void fab_stage_kv(threadgroup half * dst, device const char * base, uint64_t nb_row, int r0, int rows, int n, int Dx,
        ushort tid) {
    for (int r = 0; r < rows; ++r) {
        device const char * row = base + (r0 + r)*nb_row;
        const bool in = r0 + r < n;
        for (int c = tid; c < Dx; c += FAB_NSG*32) {
            dst[r*Dx + c] = in ? (half) fa_back_elem<T>(row, c) : (half) 0.0f;
        }
    }
}

static inline void fab_stage_rows(threadgroup half * dst, device const char * base, uint64_t nb_row, int r0, int n, int Dx,
        ushort tid, float scale = 1.0f) {
    for (int r = 0; r < FAB_BR; ++r) {
        device const float * row = (device const float *) (base + (r0 + r)*nb_row);
        const bool in = r0 + r < n;
        for (int c = tid; c < Dx; c += FAB_NSG*32) {
            dst[r*Dx + c] = in ? (half) (row[c]*scale) : (half) 0.0f;
        }
    }
}

// S (sg 0, 1: keys 0-7, 8-15) and dP (sg 2, 3) for one block into ss / sdp [FAB_BR][FAB_BC]
static inline void fab_scores(threadgroup const half * sq, threadgroup const half * sdo, threadgroup const half * sk,
        threadgroup const half * sv, threadgroup float * ss, threadgroup float * sdp, int D, ushort sgitg) {
    const bool dp = sgitg >= 2;
    const int  t  = sgitg & 1;
    threadgroup const half * a = dp ? sdo : sq;
    threadgroup const half * b = (dp ? sv : sk) + 8*t*D;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_half8x8 ma, mb;
    for (int d0 = 0; d0 < D; d0 += 8) {
        simdgroup_load(ma, a + d0, D);
        simdgroup_load(mb, b + d0, D, 0, true); // K^T: [d][key]
        simdgroup_multiply_accumulate(acc, ma, mb, acc);
    }
    simdgroup_store(acc, (dp ? sdp : ss) + 8*t, FAB_BC);
}

template <typename T>
kernel void kernel_flash_attn_back_stats(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * d,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const float ds = ((device const float *) (dst + args.offs_lse))[2*args.N*args.H]; // the dO scale
    const int D  = args.D;
    const int h  = tgpig.y;
    const int i0 = tgpig.x*FAB_BR;
    const int hk = h / (args.H / args.Hkv);

    threadgroup half  * sq  = (threadgroup half  *) shmem;
    threadgroup half  * sdo = sq  + FAB_BR*D;
    threadgroup half  * sk  = sdo + FAB_BR*D;
    threadgroup half  * sv  = sk  + FAB_BC*D;
    threadgroup float * ss  = (threadgroup float *) (sv + FAB_BC*D);
    threadgroup float * sdp = ss + FAB_BR*FAB_BC;

    fab_stage_rows(sq,  q + h*args.nbq2, args.nbq1, i0, args.N, D, tiitg);
    fab_stage_rows(sdo, d + h*args.nbd1, args.nbd2, i0, args.N, D, tiitg, ds);

    float m = -INFINITY, l = 0.0f, a = 0.0f; // thread r < FAB_BR owns row i0 + r
    for (int j0 = 0; j0 < args.M; j0 += FAB_BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fab_stage_kv<T>(sk, k + hk*args.nbk2, args.nbk1, j0, FAB_BC, args.M, D, tiitg);
        fab_stage_kv<T>(sv, v + hk*args.nbv2, args.nbv1, j0, FAB_BC, args.M, D, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fab_scores(sq, sdo, sk, sv, ss, sdp, D, sgitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < FAB_BR && i0 + tiitg < args.N) {
            const int i = i0 + tiitg;
            for (int c = 0; c < FAB_BC && j0 + c < args.M; ++c) {
                const float sc = ss[tiitg*FAB_BC + c]*args.scale + fa_back_mask(args, mask, j0 + c, i, h);
                if (sc == -INFINITY) {
                    continue;
                }
                const float mn = max(m, sc);
                const float r  = exp(m - mn);
                const float e  = exp(sc - mn);
                l = l*r + e;
                a = a*r + e*sdp[tiitg*FAB_BC + c];
                m = mn;
            }
        }
    }
    if (tiitg < FAB_BR && i0 + tiitg < args.N) {
        device float * lsev = (device float *) (dst + args.offs_lse);
        lsev[h*args.N + i0 + tiitg]               = m == -INFINITY ? -INFINITY : m + log(l);
        lsev[args.N*args.H + h*args.N + i0 + tiitg] = m == -INFINITY ? 0.0f : a/l/ds;
    }
}

// P and dS of one block from ss / sdp into half sp / sds [FAB_BR][FAB_BC]
static inline void fab_probs(constant ggml_metal_kargs_flash_attn_back & args, device const char * mask,
        threadgroup const float * ss, threadgroup const float * sdp, threadgroup const float * slse,
        threadgroup half * sp, threadgroup half * sds, int i0, int j0, int h, ushort tiitg) {
    const int r = tiitg / FAB_BC;
    const int c = tiitg % FAB_BC;
    const int i = i0 + r;
    const int j = j0 + c;
    float p = 0.0f;
    float ds = 0.0f;
    if (i < args.N && j < args.M && slse[r] != -INFINITY) {
        const float sc = ss[r*FAB_BC + c]*args.scale + fa_back_mask(args, mask, j, i, h);
        p  = sc == -INFINITY ? 0.0f : exp(sc - slse[r]);
        ds = p*(sdp[r*FAB_BC + c] - slse[FAB_BR + r])*args.scale;
    }
    sp [r*FAB_BC + c] = (half) p;
    sds[r*FAB_BC + c] = (half) ds;
}

template <typename T>
kernel void kernel_flash_attn_back_dkv(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * d,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const float ds = ((device const float *) (dst + args.offs_lse))[2*args.N*args.H]; // the dO scale
    const int D  = args.D;
    const int hk = tgpig.y;
    const int j0 = args.kv0 + tgpig.x*FAB_BC;
    const int G  = args.H / args.Hkv;
    const int CW = D / FAB_NSG; // the columns this simdgroup owns
    const int c0 = sgitg*CW;

    threadgroup half  * sq   = (threadgroup half  *) shmem;
    threadgroup half  * sdo  = sq  + FAB_BR*D;
    threadgroup half  * sk   = sdo + FAB_BR*D;
    threadgroup half  * sv   = sk  + FAB_BC*D;
    threadgroup float * ss   = (threadgroup float *) (sv + FAB_BC*D);
    threadgroup float * sdp  = ss  + FAB_BR*FAB_BC;
    threadgroup float * slse = sdp + FAB_BR*FAB_BC;          // [lse x FAB_BR][D x FAB_BR]
    threadgroup half  * sp   = (threadgroup half *) (slse + 2*FAB_BR);
    threadgroup half  * sds  = sp + FAB_BR*FAB_BC;

    fab_stage_kv<T>(sk, k + hk*args.nbk2, args.nbk1, j0, FAB_BC, args.M, D, tiitg);
    fab_stage_kv<T>(sv, v + hk*args.nbv2, args.nbv1, j0, FAB_BC, args.M, D, tiitg);

    simdgroup_float8x8 gk[2][FA_BACK_MAX_D/8/FAB_NSG];
    simdgroup_float8x8 gv[2][FA_BACK_MAX_D/8/FAB_NSG];
    for (int r = 0; r < 2; ++r) {
        for (int t = 0; t < CW/8; ++t) {
            gk[r][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            gv[r][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }
    device const float * lsev = (device const float *) (dst + args.offs_lse);
    for (int h = hk*G; h < (hk + 1)*G; ++h) {
        for (int i0 = 0; i0 < args.N; i0 += FAB_BR) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fab_stage_rows(sq,  q + h*args.nbq2, args.nbq1, i0, args.N, D, tiitg);
            fab_stage_rows(sdo, d + h*args.nbd1, args.nbd2, i0, args.N, D, tiitg, ds);
            if (tiitg < FAB_BR) {
                const int i = i0 + tiitg;
                slse[tiitg]          = i < args.N ? lsev[h*args.N + i] : -INFINITY;
                slse[FAB_BR + tiitg] = i < args.N ? lsev[args.N*args.H + h*args.N + i]*ds : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fab_scores(sq, sdo, sk, sv, ss, sdp, D, sgitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fab_probs(args, mask, ss, sdp, slse, sp, sds, i0, j0, h, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // dV[16 x CW] += P^T[16 x 8] dO[8 x CW];  dK[16 x CW] += dS^T[16 x 8] Q[8 x CW]
            simdgroup_half8x8 pt, dst8, mo, mq;
            for (int r = 0; r < 2; ++r) {
                simdgroup_load(pt,   sp  + 8*r, FAB_BC, 0, true);
                simdgroup_load(dst8, sds + 8*r, FAB_BC, 0, true);
                for (int t = 0; t < CW/8; ++t) {
                    simdgroup_load(mo, sdo + c0 + 8*t, D);
                    simdgroup_load(mq, sq  + c0 + 8*t, D);
                    simdgroup_multiply_accumulate(gv[r][t], pt,   mo, gv[r][t]);
                    simdgroup_multiply_accumulate(gk[r][t], dst8, mq, gk[r][t]);
                }
            }
        }
    }
    // through threadgroup memory (the staged K/V are done with), so a partial block writes only its rows
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * so = (threadgroup float *) sk; // [FAB_BC][D]
    for (int part = 0; part < 2; ++part) {
        for (int r = 0; r < 2; ++r) {
            for (int t = 0; t < CW/8; ++t) {
                simdgroup_store(part == 0 ? gk[r][t] : gv[r][t], so + 8*r*D + c0 + 8*t, D);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device float * out = (device float *) (dst + (part == 0 ? args.offs_dk : args.offs_dv)) + (hk*(args.M - args.kv0) + j0 - args.kv0)*D;
        for (int e = tiitg; e < FAB_BC*D; e += FAB_NSG*32) {
            if (j0 + e / D < args.M) {
                out[e] = so[e]/ds;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename T>
kernel void kernel_flash_attn_back_dq(
        constant ggml_metal_kargs_flash_attn_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * d,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const float ds = ((device const float *) (dst + args.offs_lse))[2*args.N*args.H]; // the dO scale
    const int D  = args.D;
    const int h  = tgpig.y;
    const int i0 = tgpig.x*FAB_BR;
    const int hk = h / (args.H / args.Hkv);
    const int CW = D / FAB_NSG;
    const int c0 = sgitg*CW;

    threadgroup half  * sq   = (threadgroup half  *) shmem;
    threadgroup half  * sdo  = sq  + FAB_BR*D;
    threadgroup half  * sk   = sdo + FAB_BR*D;
    threadgroup half  * sv   = sk  + FAB_BC*D;
    threadgroup float * ss   = (threadgroup float *) (sv + FAB_BC*D);
    threadgroup float * sdp  = ss  + FAB_BR*FAB_BC;
    threadgroup float * slse = sdp + FAB_BR*FAB_BC;
    threadgroup half  * sp   = (threadgroup half *) (slse + 2*FAB_BR);
    threadgroup half  * sds  = sp + FAB_BR*FAB_BC;

    fab_stage_rows(sq,  q + h*args.nbq2, args.nbq1, i0, args.N, D, tiitg);
    fab_stage_rows(sdo, d + h*args.nbd1, args.nbd2, i0, args.N, D, tiitg, ds);
    device const float * lsev = (device const float *) (dst + args.offs_lse);
    if (tiitg < FAB_BR) {
        const int i = i0 + tiitg;
        slse[tiitg]          = i < args.N ? lsev[h*args.N + i] : -INFINITY;
        slse[FAB_BR + tiitg] = i < args.N ? lsev[args.N*args.H + h*args.N + i]*ds : 0.0f;
    }

    simdgroup_float8x8 gq[FA_BACK_MAX_D/8/FAB_NSG];
    for (int t = 0; t < CW/8; ++t) {
        gq[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    for (int j0 = 0; j0 < args.M; j0 += FAB_BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fab_stage_kv<T>(sk, k + hk*args.nbk2, args.nbk1, j0, FAB_BC, args.M, D, tiitg);
        fab_stage_kv<T>(sv, v + hk*args.nbv2, args.nbv1, j0, FAB_BC, args.M, D, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fab_scores(sq, sdo, sk, sv, ss, sdp, D, sgitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        fab_probs(args, mask, ss, sdp, slse, sp, sds, i0, j0, h, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dQ[8 x CW] += dS[8 x 16] K[16 x CW]
        simdgroup_half8x8 ms, mk;
        for (int kk = 0; kk < 2; ++kk) {
            simdgroup_load(ms, sds + 8*kk, FAB_BC);
            for (int t = 0; t < CW/8; ++t) {
                simdgroup_load(mk, sk + 8*kk*D + c0 + 8*t, D);
                simdgroup_multiply_accumulate(gq[t], ms, mk, gq[t]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * so = (threadgroup float *) sk; // [FAB_BR][D]
    for (int t = 0; t < CW/8; ++t) {
        simdgroup_store(gq[t], so + c0 + 8*t, D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * out = (device float *) dst + (h*args.N + i0)*D;
    for (int e = tiitg; e < FAB_BR*D; e += FAB_NSG*32) {
        if (i0 + e / D < args.N) {
            out[e] = so[e]/ds;
        }
    }
}

typedef decltype(kernel_flash_attn_back_stats<float>) kernel_flash_attn_back_tiled_t;

template [[host_name("kernel_flash_attn_back_stats_f32")]]  kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_stats<float>;
template [[host_name("kernel_flash_attn_back_stats_f16")]]  kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_stats<half>;
template [[host_name("kernel_flash_attn_back_stats_q8_0")]] kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_stats<block_q8_0>;
template [[host_name("kernel_flash_attn_back_dkv_f32")]]    kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_dkv<float>;
template [[host_name("kernel_flash_attn_back_dkv_f16")]]    kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_dkv<half>;
template [[host_name("kernel_flash_attn_back_dkv_q8_0")]]   kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_dkv<block_q8_0>;
template [[host_name("kernel_flash_attn_back_dq_f32")]]     kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_dq<float>;
template [[host_name("kernel_flash_attn_back_dq_f16")]]     kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_dq<half>;
template [[host_name("kernel_flash_attn_back_dq_q8_0")]]    kernel kernel_flash_attn_back_tiled_t kernel_flash_attn_back_dq<block_q8_0>;

typedef decltype(kernel_flash_attn_back_q<float>)  kernel_flash_attn_back_q_t;
typedef decltype(kernel_flash_attn_back_kv<float>) kernel_flash_attn_back_kv_t;

template [[host_name("kernel_flash_attn_back_q_f32")]]   kernel kernel_flash_attn_back_q_t  kernel_flash_attn_back_q<float>;
template [[host_name("kernel_flash_attn_back_q_f16")]]   kernel kernel_flash_attn_back_q_t  kernel_flash_attn_back_q<half>;
template [[host_name("kernel_flash_attn_back_q_q8_0")]]  kernel kernel_flash_attn_back_q_t  kernel_flash_attn_back_q<block_q8_0>;
template [[host_name("kernel_flash_attn_back_kv_f32")]]  kernel kernel_flash_attn_back_kv_t kernel_flash_attn_back_kv<float>;
template [[host_name("kernel_flash_attn_back_kv_f16")]]  kernel kernel_flash_attn_back_kv_t kernel_flash_attn_back_kv<half>;
template [[host_name("kernel_flash_attn_back_kv_q8_0")]] kernel kernel_flash_attn_back_kv_t kernel_flash_attn_back_kv<block_q8_0>;
