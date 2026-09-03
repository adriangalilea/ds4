/* Standalone byte-exact regression for the pre-M5 indexer prefill scorers.
 *
 * Every optimized variant is compared directly with the legacy scorer, not
 * merely with another candidate. Cases cover the 32-token activation edge,
 * token and compressed-row tile edges, resumed unaligned positions, causal
 * masking, and a 64K-context-sized compressed frontier. The output tensor is
 * poisoned differently before each dispatch so an unwritten cell cannot pass
 * by retaining an earlier result.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ds4_gpu.h"

typedef struct {
    const char *name;
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t pos0;
} scorer_case;

typedef enum {
    SCORER_LEGACY,
    SCORER_TILED2,
    SCORER_TILED4,
    SCORER_TILED5,
} scorer_variant;

static const char *variant_name(scorer_variant variant) {
    switch (variant) {
    case SCORER_LEGACY: return "legacy";
    case SCORER_TILED2: return "tiled2";
    case SCORER_TILED4: return "tiled4";
    case SCORER_TILED5: return "tiled5";
    }
    return "unknown";
}

static int select_variant(scorer_variant variant) {
    int ok = 1;
    if (variant == SCORER_LEGACY) {
        ok &= setenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED2", "1", 1) == 0;
    } else {
        ok &= unsetenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED2") == 0;
    }
    if (variant == SCORER_TILED4) {
        ok &= setenv("DS4_METAL_INDEXER_SCORES_TILED4", "1", 1) == 0;
    } else {
        ok &= unsetenv("DS4_METAL_INDEXER_SCORES_TILED4") == 0;
    }
    if (variant == SCORER_TILED2) {
        ok &= setenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED5", "1", 1) == 0;
    } else {
        ok &= unsetenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED5") == 0;
    }
    return ok;
}

static uint32_t next_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static void fill_values(float *values, size_t count, uint32_t *state, float divisor) {
    for (size_t i = 0; i < count; i++) {
        const int32_t centered = (int32_t)(next_random(state) % 2001u) - 1000;
        values[i] = (float)centered / divisor;
    }
}

static int validate_causal_shape(
        const scorer_case *tc,
        const float       *scores,
        uint32_t           ratio) {
    for (uint32_t token = 0; token < tc->n_tokens; token++) {
        uint32_t visible = (tc->pos0 + token + 1u) / ratio;
        if (visible > tc->n_comp) visible = tc->n_comp;
        for (uint32_t comp = 0; comp < tc->n_comp; comp++) {
            const float value = scores[(size_t)token * tc->n_comp + comp];
            uint32_t bits = 0;
            memcpy(&bits, &value, sizeof(bits));
            if (comp < visible) {
                if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
                    return 0;
                }
            } else if (bits != UINT32_C(0xff800000)) {
                return 0;
            }
        }
    }
    return 1;
}

static int run_case(const scorer_case *tc) {
    const uint32_t n_head = 64u;
    const uint32_t head_dim = 128u;
    const uint32_t ratio = 4u;
    const float scale = 1.0f / 11.3137f;
    const size_t q_count = (size_t)tc->n_tokens * n_head * head_dim;
    const size_t w_count = (size_t)tc->n_tokens * n_head;
    const size_t k_count = (size_t)tc->n_comp * head_dim;
    const size_t s_count = (size_t)tc->n_tokens * tc->n_comp;
    float *hq = NULL, *hw = NULL, *hk = NULL, *reference = NULL, *observed = NULL;
    ds4_gpu_tensor *q = NULL, *w = NULL, *k = NULL, *s = NULL;
    int rc = 1;

    hq = malloc(q_count * sizeof(*hq));
    hw = malloc(w_count * sizeof(*hw));
    hk = malloc(k_count * sizeof(*hk));
    reference = malloc(s_count * sizeof(*reference));
    observed = malloc(s_count * sizeof(*observed));
    if (!hq || !hw || !hk || !reference || !observed) {
        fprintf(stderr, "FAIL %s: host allocation\n", tc->name);
        goto done;
    }

    uint32_t random_state = UINT32_C(0x9e3779b9) ^ tc->n_tokens ^ tc->n_comp ^ tc->pos0;
    fill_values(hq, q_count, &random_state, 512.0f);
    fill_values(hw, w_count, &random_state, 1024.0f);
    fill_values(hk, k_count, &random_state, 512.0f);

    q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    w = ds4_gpu_tensor_alloc(w_count * sizeof(float));
    k = ds4_gpu_tensor_alloc(k_count * sizeof(float));
    s = ds4_gpu_tensor_alloc(s_count * sizeof(float));
    if (!q || !w || !k || !s ||
        !ds4_gpu_tensor_write(q, 0, hq, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(w, 0, hw, w_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(k, 0, hk, k_count * sizeof(float))) {
        fprintf(stderr, "FAIL %s: GPU allocation/write\n", tc->name);
        goto done;
    }

    for (scorer_variant variant = SCORER_LEGACY;
         variant <= SCORER_TILED5;
         variant++) {
        const unsigned char poison = (unsigned char)(0x31u + 0x21u * variant);
        memset(observed, poison, s_count * sizeof(float));
        if (!select_variant(variant) ||
            !ds4_gpu_tensor_write(s, 0, observed, s_count * sizeof(float)) ||
            !ds4_gpu_indexer_scores_decode_batch_tensor(
                s, q, w, k, tc->n_comp, tc->n_tokens, tc->pos0,
                n_head, head_dim, ratio, scale) ||
            !ds4_gpu_tensor_read(s, 0, observed, s_count * sizeof(float))) {
            fprintf(stderr, "FAIL %s: %s dispatch/read\n",
                    tc->name, variant_name(variant));
            goto done;
        }
        if (!validate_causal_shape(tc, observed, ratio)) {
            fprintf(stderr, "FAIL %s: %s causal/output shape\n",
                    tc->name, variant_name(variant));
            goto done;
        }
        if (variant == SCORER_LEGACY) {
            memcpy(reference, observed, s_count * sizeof(float));
            continue;
        }
        if (memcmp(reference, observed, s_count * sizeof(float)) != 0) {
            size_t first = 0, differing = 0;
            for (size_t i = 0; i < s_count; i++) {
                if (memcmp(&reference[i], &observed[i], sizeof(float)) != 0) {
                    if (differing++ == 0) first = i;
                }
            }
            fprintf(stderr,
                    "FAIL %s: %s differs from legacy at %zu/%zu floats; "
                    "first token=%zu comp=%zu legacy=%a observed=%a\n",
                    tc->name, variant_name(variant), differing, s_count,
                    first / tc->n_comp, first % tc->n_comp,
                    reference[first], observed[first]);
            goto done;
        }
    }

    fprintf(stderr,
            "PASS %s: legacy/tiled2/tiled4/tiled5 identical (%zu floats)\n",
            tc->name, s_count);
    rc = 0;

done:
    ds4_gpu_tensor_free(s);
    ds4_gpu_tensor_free(k);
    ds4_gpu_tensor_free(w);
    ds4_gpu_tensor_free(q);
    free(observed);
    free(reference);
    free(hk);
    free(hw);
    free(hq);
    return rc;
}

int main(void) {
    static const scorer_case cases[] = {
        {"below-gate", 31u, 63u, 0u},
        {"activation-edge", 32u, 64u, 0u},
        {"token-and-comp-edge", 33u, 65u, 251u},
        {"unaligned-resume", 61u, 197u, 731u},
        {"64k-causal-frontier", 64u, 16387u, 4093u},
        {"64k-visible-frontier", 65u, 16387u, 65531u},
    };

    if (!ds4_gpu_init()) {
        fprintf(stderr, "FAIL: GPU init\n");
        return 1;
    }
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (run_case(&cases[i]) != 0) {
            ds4_gpu_cleanup();
            return 1;
        }
    }
    select_variant(SCORER_LEGACY);
    ds4_gpu_cleanup();

    /* Exercise init/cleanup again after the candidate scratch buffers existed. */
    if (!ds4_gpu_init() || run_case(&cases[1]) != 0) {
        fprintf(stderr, "FAIL: scorer lifecycle reinitialization\n");
        ds4_gpu_cleanup();
        return 1;
    }
    ds4_gpu_cleanup();
    unsetenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED2");
    unsetenv("DS4_METAL_INDEXER_SCORES_TILED4");
    unsetenv("DS4_METAL_DISABLE_INDEXER_SCORES_TILED5");
    fprintf(stderr, "PASS: all indexer scorer regressions\n");
    return 0;
}
