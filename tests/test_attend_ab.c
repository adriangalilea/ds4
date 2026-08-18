/* Standalone harness for the indexed mixed attention prefill path.
 * Runs without a model: random Q / raw ring / compressed cache / top-k
 * lists, output through the public GPU entry (which includes the
 * chronological top-k sort), compared against a CPU reference that
 * replicates the heads8 kernel's math exactly (f16 casts on staged K/V and
 * Q, row order, visibility cut, raw window, attention sink).
 *
 * Two uses:
 *   - default: report max/rms relative error of the GPU kernel vs the CPU
 *     reference (heads8 lands at f16-rounding noise; a candidate kernel is
 *     acceptable when its error is the same magnitude).
 *   - ATTEND_CAND_ENV=NAME: also run with that env set and report both, so
 *     an output-changing candidate can be judged against the established
 *     kernel's own noise floor.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ds4_gpu.h"

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while (0)

static float h(float v) { _Float16 x = (_Float16)v; return (float)x; }

typedef struct { double max_rel; double rms; double worst_abs; } errstat;

static void cpu_reference(
        const float *q, const float *raw, const float *comp,
        const int32_t *topk, const float *sinks, float *out,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, uint32_t n_comp, uint32_t top_k, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    int32_t *sorted = malloc(top_k * sizeof(int32_t));
    for (uint32_t t = 0; t < n_tokens; t++) {
        /* chronological sort of the token's top-k list (engine sorts
         * ascending; negatives sort first and are skipped by the scan) */
        memcpy(sorted, topk + (size_t)t * top_k, top_k * sizeof(int32_t));
        for (uint32_t i = 1; i < top_k; i++) {
            int32_t v = sorted[i];
            uint32_t j = i;
            while (j > 0 && sorted[j - 1] > v) { sorted[j] = sorted[j - 1]; j--; }
            sorted[j] = v;
        }
        const uint32_t qpos = pos0 + t;
        const uint32_t last_pos = pos0 + n_tokens - 1u;
        const uint32_t first_raw_pos = last_pos + 1u - n_raw;
        const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
        const uint32_t window_first =
            (window != 0u && qpos + 1u > window) ? qpos + 1u - window : 0u;
        const uint32_t first =
            first_raw_pos > window_first ? first_raw_pos : window_first;
        const uint32_t last = qpos < raw_last_pos ? qpos : raw_last_pos;
        for (uint32_t head = 0; head < n_head; head++) {
            const float *qh = q + ((size_t)t * n_head + head) * head_dim;
            float M = -3.4028235e38f / 2.0f;  /* -FLT_MAX/2, as the kernel */
            float S = 0.0f;
            float *o = out + ((size_t)t * n_head + head) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) o[d] = 0.0f;

            /* raw sliding window */
            if (first <= last) {
                for (uint32_t pos = first; pos <= last; pos++) {
                    const uint32_t logical = pos - first_raw_pos;
                    const uint32_t row = (raw_start + logical) % raw_cap;
                    const float *kv = raw + (size_t)row * head_dim;
                    float dots[4] = {0, 0, 0, 0};
                    for (uint32_t d = 0; d < head_dim; d++)
                        dots[(d / 4u) & 3u] += h(qh[d]) * h(kv[d]);
                    float score = (dots[0] + dots[1] + dots[2] + dots[3]) * scale;
                    const float old_m = M;
                    const float new_m = score > M ? score : M;
                    const float old_scale = expf(old_m - new_m);
                    const float row_scale = expf(score - new_m);
                    S = S * old_scale + row_scale;
                    for (uint32_t d = 0; d < head_dim; d++)
                        o[d] = o[d] * old_scale + h(kv[d]) * row_scale;
                    M = new_m;
                }
            }

            /* selected compressed rows, chronological, visibility cut */
            uint32_t visible = (qpos + 1u) / ratio;
            if (visible > n_comp) visible = n_comp;
            for (uint32_t i = 0; i < top_k; i++) {
                const int32_t idx = sorted[i];
                if (idx < 0) continue;
                if ((uint32_t)idx >= visible) break;
                const float *kv = comp + (size_t)idx * head_dim;
                float dots[4] = {0, 0, 0, 0};
                for (uint32_t d = 0; d < head_dim; d++)
                    dots[(d / 4u) & 3u] += h(qh[d]) * h(kv[d]);
                float score = (dots[0] + dots[1] + dots[2] + dots[3]) * scale;
                const float old_m = M;
                const float new_m = score > M ? score : M;
                const float old_scale = expf(old_m - new_m);
                const float row_scale = expf(score - new_m);
                S = S * old_scale + row_scale;
                for (uint32_t d = 0; d < head_dim; d++)
                    o[d] = o[d] * old_scale + h(kv[d]) * row_scale;
                M = new_m;
            }

            /* attention sink */
            {
                const float score = sinks[head];
                const float old_m = M;
                const float new_m = score > M ? score : M;
                const float old_scale = expf(old_m - new_m);
                const float row_scale = expf(score - new_m);
                S = S * old_scale + row_scale;
                for (uint32_t d = 0; d < head_dim; d++) o[d] *= old_scale;
                M = new_m;
            }

            const float inv_s = S == 0.0f ? 0.0f : 1.0f / S;
            for (uint32_t d = 0; d < head_dim; d++) o[d] *= inv_s;
        }
    }
    free(sorted);
}

static errstat compare(const float *a, const float *ref, size_t n) {
    errstat e = {0, 0, 0};
    double sq = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)ref[i]);
        const double r = d / (fabs((double)ref[i]) + 1e-6);
        if (r > e.max_rel) { e.max_rel = r; e.worst_abs = d; }
        sq += r * r;
    }
    e.rms = sqrt(sq / (double)n);
    return e;
}

int main(void) {
    CHECK(ds4_gpu_init(), "gpu init");

    const uint32_t n_tokens = 48u, pos0 = 4000u;
    const uint32_t n_raw = 128u, raw_cap = 128u, raw_start = 17u;
    const uint32_t n_comp = 1024u, top_k = 512u;
    const uint32_t window = 128u, ratio = 4u;
    const uint32_t n_head = 64u, head_dim = 512u;

    const size_t qc = (size_t)n_tokens * n_head * head_dim;
    const size_t rawc = (size_t)raw_cap * head_dim;
    const size_t compc = (size_t)n_comp * head_dim;
    const size_t tkc = (size_t)n_tokens * top_k;

    float *hq = malloc(qc * 4);
    float *hraw = malloc(rawc * 4);
    float *hcomp = malloc(compc * 4);
    int32_t *htk = malloc(tkc * 4);
    float *ref = malloc(qc * 4);
    float *got = malloc(qc * 4);
    CHECK(hq && hraw && hcomp && htk && ref && got, "host alloc");

    srand(4242);
    for (size_t i = 0; i < qc; i++) hq[i] = (float)(rand() % 2001 - 1000) / 2048.0f;
    for (size_t i = 0; i < rawc; i++) hraw[i] = (float)(rand() % 2001 - 1000) / 1024.0f;
    for (size_t i = 0; i < compc; i++) hcomp[i] = (float)(rand() % 2001 - 1000) / 1024.0f;
    /* top-k lists: unsorted unique-ish indices below each token's visibility,
     * some -1 gaps, tail beyond visibility to exercise the break */
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t visible = (pos0 + t + 1u) / ratio < n_comp ?
            (pos0 + t + 1u) / ratio : n_comp;
        for (uint32_t i = 0; i < top_k; i++) {
            int32_t v = (int32_t)((uint32_t)rand() % n_comp);
            if ((i % 37u) == 5u) v = -1;
            htk[(size_t)t * top_k + i] = v;
            (void)visible;
        }
    }

    /* fake model map: page-aligned, sinks at offset 0 */
    void *model_map = NULL;
    CHECK(posix_memalign(&model_map, 16384, 16384) == 0, "model map alloc");
    float *sinks = (float *)model_map;
    for (uint32_t i = 0; i < n_head; i++)
        sinks[i] = (float)((int)(i % 13u) - 6) / 4.0f;
    CHECK(ds4_gpu_set_model_map(model_map, 16384), "register model map");

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(qc * 4);
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(rawc * 4);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(compc * 4);
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc(tkc * 4);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(qc * 4);
    CHECK(q && raw && comp && tk && heads, "gpu alloc");
    CHECK(ds4_gpu_tensor_write(q, 0, hq, qc * 4), "wq");
    CHECK(ds4_gpu_tensor_write(raw, 0, hraw, rawc * 4), "wraw");
    CHECK(ds4_gpu_tensor_write(comp, 0, hcomp, compc * 4), "wcomp");
    CHECK(ds4_gpu_tensor_write(tk, 0, htk, tkc * 4), "wtk");

    cpu_reference(hq, hraw, hcomp, htk, sinks, ref,
                  n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                  window, ratio, n_head, head_dim);

    const char *cand = getenv("ATTEND_CAND_ENV");
    for (int pass = 0; pass < (cand ? 2 : 1); pass++) {
        if (cand) {
            if (pass == 0) unsetenv(cand);
            else setenv(cand, getenv("ATTEND_CAND_VAL") ?
                        getenv("ATTEND_CAND_VAL") : "1", 1);
        }
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                  heads, model_map, 16384, 0, q, raw, comp, 0, tk,
                  n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                  window, ratio, n_head, head_dim), "attend compute");
        CHECK(ds4_gpu_tensor_read(heads, 0, got, qc * 4), "read");
        const errstat e = compare(got, ref, qc);
        fprintf(stderr,
                "%s vs cpu-ref: max_rel=%.3e rms_rel=%.3e worst_abs=%.3e\n",
                pass == 0 ? "baseline " : "candidate", e.max_rel, e.rms,
                e.worst_abs);
    }
    ds4_gpu_cleanup();
    return 0;
}
