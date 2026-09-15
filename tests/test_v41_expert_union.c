#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#define D 5120u
#define F 2304u
#define E 384u
#define K 6u
#define R 8u

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static uint32_t state = 1937;
static uint32_t random_u32(void) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
static uint64_t aligned(uint64_t n, uint64_t page) { return (n + page - 1) / page * page; }

int main(void) {
    const uint64_t page = getpagesize(), row = D / 32 * 17, down_row = F / 32 * 17;
    const uint64_t expert = row * F, down_expert = down_row * D;
    const uint64_t up_offset = aligned(E * expert, page);
    const uint64_t down_offset = aligned(up_offset + E * expert, page);
    const uint64_t size = aligned(down_offset + E * down_expert, page);
    uint8_t *model = NULL;
    CHECK(posix_memalign((void **)&model, page, size) == 0);
    memset(model, 0, size);
    const uint64_t offsets[] = {0, up_offset, down_offset};
    for (unsigned m = 0; m < 3; ++m) {
        const uint64_t bytes = E * (m == 2 ? down_expert : expert);
        for (uint64_t b = 0; b < bytes; b += 17) {
            model[offsets[m] + b] = 115 + random_u32() % 9;
            for (unsigned q = 1; q < 17; q += 4) {
                uint32_t value = random_u32();
                memcpy(model + offsets[m] + b + q, &value, 4);
            }
        }
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model, size));
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    float x[R * D], weights[R * K];
    int32_t ids[R * K];
    for (unsigned i = 0; i < R * D; ++i)
        x[i] = ldexpf(((int)(random_u32() % 257) - 128) / 128.f,
                     (int)(random_u32() % 25) - 12);
    for (unsigned i = 0; i < R * K; ++i) weights[i] = (1 + random_u32() % 20) / 64.f;
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(sizeof(x));
    ds4_gpu_tensor *it = ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(sizeof(weights));
    ds4_gpu_tensor *t[4];
    uint64_t capacity[4] = {R*K*F, R*K*F, R*K*F, R*D};
    float *reference[4], *actual[4], *paired[4];
    CHECK(xt && it && wt);
    for (unsigned j = 0; j < 4; ++j) {
        t[j] = ds4_gpu_tensor_alloc((capacity[j] + 32) * sizeof(float));
        reference[j] = malloc((capacity[j] + 32) * sizeof(float));
        actual[j] = malloc((capacity[j] + 32) * sizeof(float));
        paired[j] = malloc((capacity[j] + 32) * sizeof(float));
        CHECK(t[j] && reference[j] && actual[j] && paired[j]);
    }
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_alloc(R*K*D*sizeof(float));
    CHECK(scratch && ds4_gpu_tensor_write(xt, 0, x, sizeof(x)) &&
          ds4_gpu_tensor_write(wt, 0, weights, sizeof(weights)));
    for (unsigned rows = 1; rows <= R; ++rows) {
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
            for (unsigned r = 0; r < rows; ++r) {
                for (unsigned k = 0; k < K; ++k) {
                    unsigned slot = (k + r) % K;
                    ids[r*K+k] = pattern == 0 ? slot :
                        pattern == 1 ? r*K+slot :
                        pattern == 2 ? (slot < 3 ? slot : 3+r*3+slot-3) :
                        E-1-((r%2)*K+slot);
                }
            }
            CHECK(ds4_gpu_tensor_write(it, 0, ids, rows*K*sizeof(int32_t)));
            for (unsigned clipped = 0; clipped < 2; ++clipped) {
                for (unsigned variant = 0; variant < 4; ++variant) {
                    if (variant) unsetenv("DS4_METAL_DISABLE_V41_EXPERT_UNION_GATE");
                    else setenv("DS4_METAL_DISABLE_V41_EXPERT_UNION_GATE", "1", 1);
                    if (variant == 1) setenv("DS4_METAL_DISABLE_V41_EXPERT_PAIR_REUSE", "1", 1);
                    else unsetenv("DS4_METAL_DISABLE_V41_EXPERT_PAIR_REUSE");
                    for (unsigned j = 0; j < 4; ++j) {
                        for (uint64_t i = 0; i < capacity[j] + 32; ++i) actual[j][i] = -12345.f;
                        CHECK(ds4_gpu_tensor_write(t[j], 0, actual[j], (capacity[j]+32)*sizeof(float)));
                    }
                    bool half = true;
                    CHECK(ds4_gpu_routed_moe_batch_tensor(t[3], t[0], t[1], t[2], scratch,
                        model, size, 0, up_offset, down_offset, 39, 39, expert, row,
                        down_expert, down_row, D, F, D, it, wt, E, K, clipped ? 7.f : 0.f,
                        xt, 0, rows, &half, true));
                    CHECK(!half);
                    for (unsigned j = 0; j < 4; ++j) {
                        const uint64_t used = rows * (j == 3 ? D : K*F);
                        CHECK(ds4_gpu_tensor_read(t[j], 0, actual[j], (capacity[j]+32)*sizeof(float)));
                        for (uint64_t i = 0; i < used; ++i) CHECK(isfinite(actual[j][i]));
                        for (uint64_t i = used; i < capacity[j]+32; ++i) CHECK(actual[j][i] == -12345.f);
                        if (!variant) memcpy(reference[j], actual[j], used*sizeof(float));
                        else if (variant == 1 && memcmp(reference[j], actual[j], used*sizeof(float))) {
                            fprintf(stderr, "MISMATCH rows=%u pattern=%u clipped=%u tensor=%u\n", rows, pattern, clipped, j);
                            return 1;
                        } else if (variant == 2) {
                            double err2 = 0, ref2 = 0, total = 0, maxerr = 0, maxref = 0;
                            for (uint64_t i = 0; i < used; ++i) {
                                const double delta = fabs((double)actual[j][i] - reference[j][i]);
                                err2 += delta * delta;
                                ref2 += (double)reference[j][i] * reference[j][i];
                                total += delta;
                                maxerr = fmax(maxerr, delta);
                                maxref = fmax(maxref, fabs(reference[j][i]));
                            }
                            const double relative_l2 = sqrt(err2 / fmax(ref2, 1e-30));
                            const double relative_max = maxerr / fmax(maxref, 1e-30);
                            fprintf(stderr, "NUMERICAL rows=%u pattern=%u clamp=%u tensor=%u max_abs=%.9g mean_abs=%.9g relative_l2=%.9g relative_max=%.9g\n",
                                    rows, pattern, clipped, j, maxerr, total / used, relative_l2, relative_max);
                            /* Wide inputs can nearly cancel before SwiGLU;
                             * allow 0.1% downstream error, 0.001% at projections. */
                            const double limit = j < 2 ? 1e-5 : 1e-3;
                            CHECK(relative_l2 <= limit && relative_max <= limit);
                            if (rows < 4) CHECK(!memcmp(reference[j], actual[j], used*sizeof(float)));
                            memcpy(paired[j], actual[j], used*sizeof(float));
                        } else if (variant == 3) {
                            CHECK(!memcmp(paired[j], actual[j], used*sizeof(float)));
                        }
                    }
                }
                fprintf(stderr, "PASS expert union rows=%u pattern=%u clamp=%u grouped_exact=1 paired_repeat_exact=1\n", rows, pattern, clipped);
            }
        }
    }
    for (unsigned j = 0; j < 4; ++j) {
        ds4_gpu_tensor_free(t[j]); free(reference[j]); free(actual[j]); free(paired[j]);
    }
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(it); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(scratch);
    ds4_gpu_cleanup();
    free(model);
    return 0;
}
