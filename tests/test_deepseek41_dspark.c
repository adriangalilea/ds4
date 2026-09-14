/* Real-weight drafter checks; only embedding/output tensors of the target are read. */
#include "../ds4.c"
#include <assert.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TARGET.gguf DSPARK.gguf\n", argv[0]);
        return 2;
    }
    int rc = 1;
    ds4_model target = {.fd = -1}, support = {.fd = -1};
    ds4_weights weights = {0};
    ds41_dspark_graph draft = {0};
    ds4_gpu_tensor *hidden = NULL, *last = NULL;
    float *features = NULL, *first_logits = NULL, *second_logits = NULL;
    model_open(&target, argv[1], true, false);
    config_validate_model(&target);
    CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&weights, &target, false, 0, UINT32_MAX, true, false);
    model_open(&support, argv[2], true, false);
    CHECK(support_model_checkpoint_compatible(&support));
    const ds4_dspark_summary summary = model_dspark_summary(&support);
    ds4_dspark_weights dw;
    dspark_weights_bind_optional(&dw, &support, &summary);
    CHECK(dw.v41 && !dw.missing_tensors && !dw.invalid_tensors && !dw.metadata_errors);
    ds4_gpu_model_residency_skip(1);
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map_range(support.map, support.size, support.tensor_data_pos,
                                      support.size - support.tensor_data_pos, support.max_tensor_bytes));
    CHECK(ds41_dspark_alloc(&draft, &dw));
    const uint32_t width = dw.target_layer_count * DS4_N_EMBD;
    const size_t feature_bytes = (size_t)dw.sliding_window * width * sizeof(float);
    const size_t logits_bytes = (size_t)dw.block_size * DS4_N_VOCAB * sizeof(float);
    hidden = ds4_gpu_tensor_alloc(feature_bytes);
    features = malloc(feature_bytes);
    first_logits = malloc(logits_bytes);
    second_logits = malloc(logits_bytes);
    CHECK(hidden && features && first_logits && second_logits);
    const uint32_t starts[] = {0, 0, 1, 129, 1025};
    const uint32_t counts[] = {2, 128, 128, 128, 128};
    for (uint32_t fixture = 0; fixture < 5; fixture++) {
        const uint32_t start = starts[fixture], count = counts[fixture];
        const uint32_t pos = start + count - 1u;
        for (uint32_t i = 0; i < count * width; i++)
            features[i] = (float)((int)((i * 37u + fixture * 17u) % 257u) - 128) / 128.0f;
        CHECK(ds4_gpu_tensor_write(hidden, 0, features, (size_t)count * width * sizeof(float)));
        CHECK(ds41_dspark_seed(&draft, &support, hidden, start, count));
        CHECK(draft.cache_end == pos + 1u);
        /* Every seeded row must occupy its absolute-position ring slot. */
        float projected[512], cached[512];
        for (uint32_t i = 0; i < count; i++) {
            CHECK(ds4_gpu_tensor_read(draft.main_kv, (uint64_t)i * sizeof(projected), projected, sizeof(projected)));
            CHECK(ds4_gpu_tensor_read(draft.window[2], (uint64_t)((start + i) % 128u) * sizeof(cached), cached, sizeof(cached)));
            CHECK(!memcmp(projected, cached, sizeof(cached)));
        }
        last = ds4_gpu_tensor_view(hidden, (uint64_t)(count - 1u) * width * sizeof(float),
                                   (uint64_t)width * sizeof(float));
        CHECK(last);
        int32_t first[16] = {0}, second[16] = {0};
        float confidence1[16] = {0}, confidence2[16] = {0};
        double elapsed = 0;
        for (uint32_t repeat = 0; repeat < 2; repeat++) {
            /* Out-of-window poison must be invisible to short-context attention. */
            if (count < 128u) {
                float poison[512];
                for (uint32_t i = 0; i < 512; i++) poison[i] = repeat ? -19.0f : 23.0f;
                for (uint32_t stage = 0; stage < 3; stage++)
                    for (uint32_t row = count; row < 128u; row++)
                        CHECK(ds4_gpu_tensor_write(draft.window[stage], (uint64_t)row * sizeof(poison), poison, sizeof(poison)));
            }
            const double t0 = now_sec();
            CHECK(ds41_dspark_forward(&draft, &target, &weights, &support, last, pos, 100,
                                        repeat ? second : first, repeat ? confidence2 : confidence1));
            elapsed = (now_sec() - t0) * 1000.0;
            CHECK(ds4_gpu_tensor_read(draft.base_logits, 0,
                                       repeat ? second_logits : first_logits, logits_bytes));
        }
        CHECK(!memcmp(first, second, dw.block_size * sizeof(int32_t)));
        CHECK(!memcmp(confidence1, confidence2, dw.block_size * sizeof(float)));
        CHECK(!memcmp(first_logits, second_logits, logits_bytes));
        for (uint32_t i = 0; i < dw.block_size; i++) {
            CHECK(first[i] >= 0 && (uint32_t)first[i] < DS4_N_VOCAB && isfinite(confidence1[i]));
            for (uint32_t j = 0; j < DS4_N_VOCAB; j++) CHECK(isfinite(first_logits[i * DS4_N_VOCAB + j]));
        }
        CHECK(!ds41_dspark_forward(&draft, &target, &weights, &support, last, pos + 2u, 100, second, confidence2));
        printf("V4.1 DSpark pos=%u: repeat exact, ring/mask/position gates PASS, %.3f ms, proposals", pos, elapsed);
        for (uint32_t i = 0; i < dw.block_size; i++) printf(" %d", first[i]);
        puts("");
        ds4_gpu_tensor_free(last); last = NULL;
    }
    puts("V4.1 DSpark real-weight Metal forward: PASS (synthetic target features; acceptance not measured)");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(last);
    ds4_gpu_tensor_free(hidden);
    ds41_dspark_free(&draft);
    ds4_gpu_cleanup();
    model_close(&support);
    model_close(&target);
    free(features); free(first_logits); free(second_logits);
    return rc;
}
