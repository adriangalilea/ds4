/* Decides the bold-path-A stream-split design: can two internally-serial
 * dependency chains overlap on the GPU when placed in separate command
 * buffers (single queue, disjoint tracked buffers), and how does that
 * compare to (a) one serial encoder and (c) one concurrent encoder with a
 * full barrier per chain step?
 *
 *   A: both chains interleaved in ONE serial encoder (today's decode shape)
 *   B: chain per COMMAND BUFFER, committed together, one wait
 *   C: both chains in ONE concurrent encoder, memoryBarrier per step
 *      (measures the full-execution-barrier serialization claim)
 *   D: chain per CB with UNTRACKED buffers + explicit single wait
 *
 * Each chain step is a bandwidth-heavy pass over its own big buffer with a
 * serial dependency on the previous step's output. If B ~= max(chain) while
 * A ~= sum, stream-split decode is viable. cc -O2 -o tests/test_cb_overlap
 * tests/test_cb_overlap.m -framework Metal -framework Foundation */
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void chain_step(device float *buf [[buffer(0)]],\n"
"                       device const float *dep [[buffer(1)]],\n"
"                       device const float *shared_ro [[buffer(3)]],\n"
"                       device float *out [[buffer(4)]],\n"
"                       constant uint &n [[buffer(2)]],\n"
"                       uint gid [[thread_position_in_grid]],\n"
"                       uint threads [[threads_per_grid]]) {\n"
"    float acc = dep[gid % 1024u] + shared_ro[gid % 1024u];\n"
"    for (uint i = gid; i < n; i += threads) acc += buf[i];\n"
"    out[gid % 1024u] = acc;\n"
"}\n";

static double ms(uint64_t a, uint64_t b) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)(b - a) * tb.numer / tb.denom / 1e6;
}

int main(int argc, char **argv) {
    const uint32_t STEPS = argc > 1 ? (uint32_t)atoi(argv[1]) : 8;
    const uint32_t MB = argc > 2 ? (uint32_t)atoi(argv[2]) : 256;
    const uint32_t GRID = argc > 3 ? (uint32_t)atoi(argv[3]) : 32768;
    const uint32_t N = MB * 1024u * 1024u / 4u;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    fprintf(stderr, "device %s  steps=%u  buf=%u MB x2 chains  grid=%u\n",
            dev.name.UTF8String, STEPS, MB, GRID);
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:SRC] options:nil error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", err.description.UTF8String); return 1; }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:
        [lib newFunctionWithName:@"chain_step"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    /* one big buffer per chain (read target) + one small dep buffer per chain */
    id<MTLBuffer> big[2], dep[2];
    for (int c = 0; c < 2; c++) {
        big[c] = [dev newBufferWithLength:(NSUInteger)N * 4 options:MTLResourceStorageModeShared];
        dep[c] = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        float *p = big[c].contents;
        for (uint32_t i = 0; i < N; i += 4096) p[i] = 1.0f;
    }
    const MTLSize grid = MTLSizeMake(GRID, 1, 1);
    const MTLSize tg = MTLSizeMake(256, 1, 1);

    id<MTLBuffer> shared_ro = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    id<MTLBuffer> big0 = big[0], big1 = big[1], dep0 = dep[0], dep1 = dep[1];
    void (^encode_step)(id<MTLComputeCommandEncoder>, int) =
        ^(id<MTLComputeCommandEncoder> e, int c) {
            [e setComputePipelineState:pso];
            [e setBuffer:(c ? big1 : big0) offset:0 atIndex:0];
            [e setBuffer:(c ? dep1 : dep0) offset:0 atIndex:1];
            [e setBytes:&N length:4 atIndex:2];
            [e setBuffer:shared_ro offset:0 atIndex:3];
            [e setBuffer:(c ? dep1 : dep0) offset:0 atIndex:4];
            [e dispatchThreads:grid threadsPerThreadgroup:tg];
        };

    for (int rep = 0; rep < 3; rep++) {
        /* A: one serial encoder, chains interleaved */
        uint64_t t0 = mach_absolute_time();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (uint32_t s = 0; s < STEPS; s++) { encode_step(e, 0); encode_step(e, 1); }
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        double a = ms(t0, mach_absolute_time());
        double a_gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;

        /* B: one CB per chain, committed together */
        t0 = mach_absolute_time();
        id<MTLCommandBuffer> cbs[2];
        for (int c = 0; c < 2; c++) {
            cbs[c] = [q commandBuffer];
            id<MTLComputeCommandEncoder> ec = [cbs[c] computeCommandEncoder];
            for (uint32_t s = 0; s < STEPS; s++) encode_step(ec, c);
            [ec endEncoding];
        }
        [cbs[0] commit]; [cbs[1] commit];
        [cbs[1] waitUntilCompleted]; [cbs[0] waitUntilCompleted];
        double b = ms(t0, mach_absolute_time());

        /* C: one concurrent encoder, full barrier between steps */
        t0 = mach_absolute_time();
        cb = [q commandBuffer];
        e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (uint32_t s = 0; s < STEPS; s++) {
            encode_step(e, 0); encode_step(e, 1);
            id<MTLResource> res[2] = { big0, big1 };
            if (s + 1 < STEPS) [e memoryBarrierWithResources:res count:2];
        }
        [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        double cc_ = ms(t0, mach_absolute_time());

        /* D: like B but with two queues (upper bound: no queue-level order) */
        static id<MTLCommandQueue> q2;
        if (!q2) q2 = [dev newCommandQueue];
        t0 = mach_absolute_time();
        id<MTLCommandQueue> qs[2] = { q, q2 };
        for (int c = 0; c < 2; c++) {
            cbs[c] = [qs[c] commandBuffer];
            id<MTLComputeCommandEncoder> ec = [cbs[c] computeCommandEncoder];
            for (uint32_t s = 0; s < STEPS; s++) encode_step(ec, c);
            [ec endEncoding];
        }
        [cbs[0] commit]; [cbs[1] commit];
        [cbs[1] waitUntilCompleted]; [cbs[0] waitUntilCompleted];
        double d = ms(t0, mach_absolute_time());

        /* E: two CBs, write targets are two OFFSETS of one shared buffer
         * (the arena/suballocation case; tracked hazards are per-resource) */
        static id<MTLBuffer> arena;
        if (!arena) arena = [dev newBufferWithLength:8192 options:MTLResourceStorageModeShared];
        t0 = mach_absolute_time();
        for (int c = 0; c < 2; c++) {
            cbs[c] = [q commandBuffer];
            id<MTLComputeCommandEncoder> ec = [cbs[c] computeCommandEncoder];
            for (uint32_t s2 = 0; s2 < STEPS; s2++) {
                [ec setComputePipelineState:pso];
                [ec setBuffer:(c ? big1 : big0) offset:0 atIndex:0];
                [ec setBuffer:(c ? dep1 : dep0) offset:0 atIndex:1];
                [ec setBytes:&N length:4 atIndex:2];
                [ec setBuffer:shared_ro offset:0 atIndex:3];
                [ec setBuffer:arena offset:(NSUInteger)c * 4096 atIndex:4];
                [ec dispatchThreads:grid threadsPerThreadgroup:tg];
            }
            [ec endEncoding];
        }
        [cbs[0] commit]; [cbs[1] commit];
        [cbs[1] waitUntilCompleted]; [cbs[0] waitUntilCompleted];
        double eo = ms(t0, mach_absolute_time());

        /* F: the decode shape — LAYERS iterations of: main CB runs one BIG
         * step (q_path), side CB runs SMALL_STEPS small steps (the
         * kv/compressor/indexer chain), cross-synchronized per layer with
         * MTLEvents exactly like the planned decode side-stream:
         *   main:  big(L)                    -> signal e_main(L) ...
         *   side:  wait e_main(L-1 join)? -- here: side chain of layer L
         *          depends on nothing from big(L) (models S2 independence
         *          from q_b), but layer L+1's big depends on side(L)'s
         *          completion (models attend needing topk):
         *   main:  big(L) ; wait side_done(L) ; big(L+1) ...
         *   side:  small*8(L) ; signal side_done(L) ; small*8(L+1) ...
         * G: same layers/steps but everything serial in one encoder
         *    (the baseline decode shape). */
        {
            const uint32_t LAYERS = argc > 4 ? (uint32_t)atoi(argv[4]) : 8;
            const uint32_t SMALL_STEPS = argc > 5 ? (uint32_t)atoi(argv[5]) : 32;
            const uint32_t SMALL_GRID = 4096;
            const uint32_t SMALL_N = argc > 6 ? (uint32_t)atoi(argv[6]) * 256u
                                              : 2u * 1024u * 1024u / 4u; /* argv[6] = KB per small step */
            static id<MTLSharedEvent> ev_shared;
            static id<MTLEvent> ev_plain;
            static uint64_t plain_base_ctr;
            const bool evplain = getenv("EVPLAIN") != NULL;
            if (!ev_shared) ev_shared = [dev newSharedEvent];
            if (!ev_plain) ev_plain = [dev newEvent];
            id<MTLEvent> ev = evplain ? ev_plain : (id<MTLEvent>)ev_shared;
            const uint64_t base = evplain ? plain_base_ctr : ev_shared.signaledValue;

            /* G baseline: serial */
            uint64_t tg0 = mach_absolute_time();
            id<MTLCommandBuffer> gcb = [q commandBuffer];
            id<MTLComputeCommandEncoder> ge = [gcb computeCommandEncoder];
            for (uint32_t L = 0; L < LAYERS; L++) {
                encode_step(ge, 0);   /* big */
                for (uint32_t s2 = 0; s2 < SMALL_STEPS; s2++) {
                    [ge setComputePipelineState:pso];
                    [ge setBuffer:big1 offset:0 atIndex:0];
                    [ge setBuffer:dep1 offset:0 atIndex:1];
                    [ge setBytes:&SMALL_N length:4 atIndex:2];
                    [ge setBuffer:shared_ro offset:0 atIndex:3];
                    [ge setBuffer:dep1 offset:0 atIndex:4];
                    [ge dispatchThreads:MTLSizeMake(SMALL_GRID,1,1) threadsPerThreadgroup:tg];
                }
            }
            [ge endEncoding];
            [gcb commit]; [gcb waitUntilCompleted];
            double gms = ms(tg0, mach_absolute_time());

            /* F: dual CB, event-pipelined */
            uint64_t tf0 = mach_absolute_time();
            static id<MTLSharedEvent> ev2_shared;
            static id<MTLEvent> ev2_plain;
            static uint64_t plain_base2_ctr;
            if (!ev2_shared) ev2_shared = [dev newSharedEvent];
            if (!ev2_plain) ev2_plain = [dev newEvent];
            id<MTLEvent> ev2 = evplain ? ev2_plain : (id<MTLEvent>)ev2_shared;
            const uint64_t base2 = evplain ? plain_base2_ctr : ev2_shared.signaledValue;
            const bool bidir = getenv("BIDIR") != NULL;
            static id<MTLCommandQueue> qside;
            if (!qside) qside = [dev newCommandQueue];
            id<MTLCommandBuffer> mainCB = [q commandBuffer];
            id<MTLCommandBuffer> sideCB = [(bidir ? qside : q) commandBuffer];
            for (uint32_t L = 0; L < LAYERS; L++) {
                /* main: big step for layer L, then wait for side(L) */
                if (bidir) [mainCB encodeSignalEvent:ev2 value:base2 + L + 1];
                id<MTLComputeCommandEncoder> me = [mainCB computeCommandEncoder];
                encode_step(me, 0);
                [me endEncoding];
                [mainCB encodeWaitForEvent:ev value:base + L + 1];
                /* side: small chain for layer L, then signal */
                if (bidir) [sideCB encodeWaitForEvent:ev2 value:base2 + L + 1];
                id<MTLComputeCommandEncoder> se = [sideCB computeCommandEncoder];
                for (uint32_t s2 = 0; s2 < SMALL_STEPS; s2++) {
                    [se setComputePipelineState:pso];
                    [se setBuffer:big1 offset:0 atIndex:0];
                    [se setBuffer:(getenv("CROSSDEP") ? dep0 : dep1) offset:0 atIndex:1];
                    [se setBytes:&SMALL_N length:4 atIndex:2];
                    [se setBuffer:shared_ro offset:0 atIndex:3];
                    [se setBuffer:dep1 offset:0 atIndex:4];
                    [se dispatchThreads:MTLSizeMake(SMALL_GRID,1,1) threadsPerThreadgroup:tg];
                }
                [se endEncoding];
                [sideCB encodeSignalEvent:ev value:base + L + 1];
            }
            [sideCB commit]; [mainCB commit];
            [mainCB waitUntilCompleted];
            double fms = ms(tf0, mach_absolute_time());
            if (evplain) {
                plain_base_ctr = base + LAYERS;
                if (bidir) plain_base2_ctr = base2 + LAYERS;
            } else {
                ev_shared.signaledValue = base + LAYERS;
                if (bidir) ev2_shared.signaledValue = base2 + LAYERS;
            }

            fprintf(stderr,
                "rep%d  A serial-enc %.1f ms (gpu %.1f)  B two-CB %.1f  C conc+barrier %.1f  D two-queue %.1f  E arena-dep %.1f  |  G layered-serial %.1f  F event-pipelined %.1f  F/G=%.2f\n",
                rep, a, a_gpu, b, cc_, d, eo, gms, fms, fms / gms);
        }
    }
    return 0;
}
