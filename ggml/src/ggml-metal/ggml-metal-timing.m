// Continuum fork: see ggml-metal-timing.h.
#import "ggml-metal-timing.h"

#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

static int timing_on(void) {
    static int on = -1;
    if (on < 0) {
        const char * v = getenv("GGML_METAL_TIMING");
        on = (v && atoi(v) != 0) ? 1 : 0;
    }
    return on;
}

static double host_seconds(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double) mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

// One graph in flight (graphs on one context run one after another).
static double encode_begin_s, encode_end_s;
static int    graph_nodes;
// The previous graph's last GPU end, on the GPU's clock (CFTimeInterval seconds).
static double last_gpu_end = 0.0;

// Rolling sums, printed every REPORT_EVERY graphs.
enum { REPORT_EVERY = 256 };
static int    n_graphs;
static double sum_encode, sum_span, sum_busy, sum_gap, max_gap;
static long   sum_nodes, sum_bufs;

void ggml_metal_timing_encode_begin(int n_nodes) {
    if (!timing_on()) {
        return;
    }
    encode_begin_s = host_seconds();
    graph_nodes = n_nodes;
}

void ggml_metal_timing_encode_end(void) {
    if (!timing_on()) {
        return;
    }
    encode_end_s = host_seconds();
}

void ggml_metal_timing_completed(const id<MTLCommandBuffer> * bufs, int n) {
    if (!timing_on()) {
        return;
    }
    double first_start = 0.0, last_end = 0.0, busy = 0.0;
    int counted = 0;
    for (int i = 0; i < n; ++i) {
        id<MTLCommandBuffer> b = bufs[i];
        if (!b || b.status != MTLCommandBufferStatusCompleted) {
            continue;
        }
        const double s = b.GPUStartTime, e = b.GPUEndTime;
        if (e <= s) {
            continue;
        }
        if (counted == 0 || s < first_start) {
            first_start = s;
        }
        if (e > last_end) {
            last_end = e;
        }
        busy += e - s;
        ++counted;
    }
    if (counted == 0) {
        return;
    }
    // The GPU sat idle between the previous graph's end and this graph's start: the round trip.
    const double gap = last_gpu_end > 0.0 && first_start > last_gpu_end ? first_start - last_gpu_end : 0.0;
    last_gpu_end = last_end;

    sum_encode += encode_end_s > encode_begin_s ? encode_end_s - encode_begin_s : 0.0;
    sum_span   += last_end - first_start;
    sum_busy   += busy;
    sum_gap    += gap;
    max_gap     = gap > max_gap ? gap : max_gap;
    sum_nodes  += graph_nodes;
    sum_bufs   += counted;
    if (++n_graphs == REPORT_EVERY) {
        const double k = 1000.0 / n_graphs;
        fprintf(stderr,
                "ggml_metal_timing: %d graphs: nodes %.0f, buffers %.1f, host encode %.3f ms, "
                "gpu span %.3f ms, gpu busy %.3f ms, idle gap before %.3f ms (max %.3f)\n",
                n_graphs, (double) sum_nodes / n_graphs, (double) sum_bufs / n_graphs,
                sum_encode * k, sum_span * k, sum_busy * k, sum_gap * k, max_gap * 1000.0);
        n_graphs = 0;
        sum_encode = sum_span = sum_busy = sum_gap = max_gap = 0.0;
        sum_nodes = sum_bufs = 0;
    }
}
