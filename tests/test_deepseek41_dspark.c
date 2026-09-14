/* Real-weight drafter checks; only embedding/output tensors of the target are read. */
#include "../ds4.c"
#include <assert.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static int check_capture(void) {
    int rc = 1;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    ds4_dspark_weights dw = {.v41 = true, .target_layer_count = 3,
        .sliding_window = 128, .target_layers = {37, 38, 39}};
    ds41_gpu_graph g = {0};
    ds4_gpu_tensor *residual = NULL;
    float *x = NULL;
    CHECK(ds4_gpu_init());
    CHECK(ds41_dspark_capture_alloc(&g, &dw));
    const size_t bytes = (size_t)150 * 4 * DS4_N_EMBD * sizeof(float);
    residual = ds4_gpu_tensor_alloc(bytes);
    x = malloc(bytes);
    CHECK(residual && x);
    for (uint32_t tap = 0; tap < 3; tap++) {
        for (uint32_t row = 0; row < 150; row++)
            for (uint32_t hc = 0; hc < 4; hc++)
                for (uint32_t col = 0; col < DS4_N_EMBD; col++)
                    x[((size_t)row * 4 + hc) * DS4_N_EMBD + col] = tap * 32 + row % 17 + col % 11 + hc;
        CHECK(ds4_gpu_tensor_write(residual, 0, x, bytes));
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds41_dspark_capture_rows(g.dspark_capture, 37 + tap, residual, 100, 150));
        CHECK(ds4_gpu_end_commands());
    }
    CHECK(ds41_dspark_capture_valid(g.dspark_capture, 122, 128));
    CHECK(!ds41_dspark_capture_valid(g.dspark_capture, 121, 128));
    float hidden[5120];
    for (uint32_t pos = 122; pos < 250; pos++)
        for (uint32_t tap = 0; tap < 3; tap++) {
            CHECK(ds4_gpu_tensor_read(g.dspark_capture->hidden,
                ((uint64_t)(pos % 128u) * 3u + tap) * sizeof(hidden), hidden, sizeof(hidden)));
            for (uint32_t col = 0; col < DS4_N_EMBD; col++)
                CHECK(hidden[col] == tap * 32 + (pos - 100) % 17 + col % 11 + 1.5f);
        }
    ds41_graph_reset(&g);
    CHECK(!ds41_dspark_capture_valid(g.dspark_capture, 122, 128));
    puts("V4.1 DSpark HC means, concatenation, ring truncation and reset: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(residual);
    ds41_dspark_capture_free(g.dspark_capture);
    ds4_gpu_cleanup();
    free(x);
    return rc;
}

static int check_live(const char *target_path, const char *support_path,
                        const char *prompt_path, bool capture) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds41_dspark_graph draft = {0};
    ds4_gpu_tensor *last = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t prompt_bytes = 0;
    ds4_engine_options opt = {.model_path = target_path, .mtp_path = support_path,
        .backend = DS4_BACKEND_METAL, .context_size = 32768, .power_percent = 100,
        .dspark = true};
    CHECK(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    CHECK(prompt_bytes && ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len > 1 && tokens.len + 256 < 32768);
    CHECK(ds4_session_create(&session, engine, 32768) == 0);
    if (capture) {
        CHECK(ds41_dspark_capture_alloc(&session->ds41_graph, &engine->dspark_weights));
        CHECK(ds41_dspark_alloc(&draft, &engine->dspark_weights));
    }
    CHECK(ds4_session_sync(session, &tokens, err, sizeof(err)) == 0);
    if (capture) CHECK(ds41_dspark_seed_capture(&draft, &engine->mtp_model,
                                                session->ds41_graph.dspark_capture, (uint32_t)tokens.len));
    int generated[256];
    int32_t proposals[256][16] = {{0}};
    float confidence[256][16] = {{0}};
    double target_ms = 0, draft_ms = 0;
    for (uint32_t i = 0; i < 256; i++) {
        const int token = sample_argmax(session->logits, DS4_N_VOCAB);
        CHECK(!vocab_token_is_generation_stop(&engine->vocab, token));
        generated[i] = token;
        size_t len = 0;
        char *piece = ds4_token_text(engine, token, &len);
        CHECK(piece);
        fwrite(piece, 1, len, stdout);
        free(piece);
        if (capture) {
            const uint32_t pos = (uint32_t)session->checkpoint.len - 1u;
            ds41_dspark_capture *c = session->ds41_graph.dspark_capture;
            CHECK(ds41_dspark_capture_valid(c, pos, 1));
            const uint64_t bytes = (uint64_t)3 * DS4_N_EMBD * sizeof(float);
            last = ds4_gpu_tensor_view(c->hidden, (pos % 128u) * bytes, bytes);
            CHECK(last);
            const double t0 = now_sec();
            CHECK(ds41_dspark_forward(&draft, &engine->model, &engine->weights, &engine->mtp_model,
                                        last, pos, token, proposals[i], confidence[i]));
            draft_ms += (now_sec() - t0) * 1000;
            ds4_gpu_tensor_free(last); last = NULL;
        }
        if (i + 1u < 256) {
            const double t0 = now_sec();
            CHECK(ds4_session_eval(session, token, err, sizeof(err)) == 0);
            target_ms += (now_sec() - t0) * 1000;
        }
    }
    fprintf(stderr, "V4.1 DSpark live prompt_tokens=%d generated=256 capture=%d target_ms=%.3f draft_ms=%.3f\n",
            tokens.len, capture, target_ms / 255, draft_ms / 256);
    fprintf(stderr, "target_tokens=");
    for (uint32_t i = 0; i < 256; i++) fprintf(stderr, "%s%d", i ? "," : "", generated[i]);
    fputc('\n', stderr);
    if (capture) {
        uint32_t prefix_total = 0, hist[6] = {0};
        for (uint32_t i = 0; i < 251; i++) {
            uint32_t n = 0;
            while (n < 5 && proposals[i][n] == generated[i + n + 1u]) n++;
            prefix_total += n; hist[n]++;
            fprintf(stderr, "draft position=%u matched=%u ids=", i, n);
            for (uint32_t j = 0; j < 5; j++)
                fprintf(stderr, "%s%d:%.6f", j ? "," : "", proposals[i][j], confidence[i][j]);
            fputc('\n', stderr);
        }
        for (unsigned prune = 0; prune < 2; prune++) {
            uint32_t cycles = 0, verified = 0, emitted = 0;
            for (uint32_t i = 0; i < 251;) {
                uint32_t cap = 0, n = 0;
                while (cap < 5 && (!prune || sigmoid_stable(confidence[i][cap]) >= .6f)) cap++;
                while (n < cap && proposals[i][n] == generated[i + n + 1u]) n++;
                cycles++; verified += cap + 1u; emitted += n + 1u; i += n + 1u;
            }
            fprintf(stderr, "V4.1 DSpark acceptance confidence=%.1f simulated_tokens_per_cycle=%.4f proposed_rows_per_cycle=%.4f cycles=%u\n",
                    prune ? .6 : 0., (double)emitted / cycles, (double)verified / cycles, cycles);
        }
        fprintf(stderr, "V4.1 DSpark prefix_mean=%.4f histogram=%u,%u,%u,%u,%u,%u (offline greedy oracle; no speculative commits)\n",
                (double)prefix_total / 251, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
    }
    rc = 0;
done:
    if (rc) fprintf(stderr, "V4.1 live drafter error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(last);
    ds41_dspark_free(&draft);
    ds4_session_free(session);
    ds4_engine_close(engine);
    ds4_tokens_free(&tokens);
    free(prompt);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--capture")) return check_capture();
    if (argc == 6 && !strcmp(argv[1], "--live") &&
        (!strcmp(argv[5], "draft") || !strcmp(argv[5], "baseline")))
        return check_live(argv[2], argv[3], argv[4], !strcmp(argv[5], "draft"));
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
    const uint64_t target_offsets[] = {weights.token_embd->abs_offset, weights.output->abs_offset};
    const uint64_t target_sizes[] = {weights.token_embd->bytes, weights.output->bytes};
    CHECK(ds4_gpu_set_model_map_spans(target.map, target.size, target_offsets, target_sizes, 2,
        target_sizes[0] > target_sizes[1] ? target_sizes[0] : target_sizes[1]));
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
