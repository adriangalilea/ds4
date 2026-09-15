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
    ds4_gpu_tensor *residual = NULL, *packed = NULL;
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
    packed = ds4_gpu_tensor_alloc((uint64_t)128 * 3 * DS4_N_EMBD * sizeof(float));
    CHECK(packed && ds41_dspark_capture_pack(packed, g.dspark_capture, 250));
    CHECK(!ds4_gpu_commands_active());
    float hidden[5120];
    for (uint32_t pos = 122; pos < 250; pos++)
        for (uint32_t tap = 0; tap < 3; tap++) {
            CHECK(ds4_gpu_tensor_read(g.dspark_capture->hidden,
                ((uint64_t)(pos % 128u) * 3u + tap) * sizeof(hidden), hidden, sizeof(hidden)));
            for (uint32_t col = 0; col < DS4_N_EMBD; col++)
                CHECK(hidden[col] == tap * 32 + (pos - 100) % 17 + col % 11 + 1.5f);
            CHECK(ds4_gpu_tensor_read(packed,
                ((uint64_t)(pos - 122) * 3u + tap) * sizeof(hidden), hidden, sizeof(hidden)));
            for (uint32_t col = 0; col < DS4_N_EMBD; col++)
                CHECK(hidden[col] == tap * 32 + (pos - 100) % 17 + col % 11 + 1.5f);
        }
    ds41_graph_reset(&g);
    CHECK(!ds41_dspark_capture_valid(g.dspark_capture, 122, 128));
    CHECK(!ds41_dspark_capture_pack(packed, g.dspark_capture, 250));
    puts("V4.1 DSpark HC means, concatenation, ring packing and reset: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(packed);
    ds41_dspark_capture_free(g.dspark_capture);
    ds4_gpu_cleanup();
    free(x);
    return rc;
}

typedef struct {
    ds4_gpu_tensor *value[4];
    uint8_t seen[4][3][384];
    uint32_t start, count;
} tap_trace;

typedef struct {
    ds4_gpu_tensor *value;
    uint8_t seen[256][40];
    uint32_t start;
} route_trace;

static bool capture_routes(void *ud, uint32_t layer, uint32_t pos,
                            const ds4_gpu_tensor *selected) {
    route_trace *t = ud;
    if (pos < t->start || pos - t->start >= 256) return true;
    if (layer >= 40 || DS4_N_EXPERT_USED != 6) return false;
    const uint32_t row = pos - t->start;
    const uint64_t bytes = 6u * sizeof(int32_t);
    if (!ds4_gpu_tensor_copy(t->value, ((uint64_t)row * 40u + layer) * bytes,
                             selected, 0, bytes)) return false;
    t->seen[row][layer] = 1;
    return true;
}

static uint64_t weight_bytes(const ds4_tensor *t) { return t ? t->bytes : 0; }

static bool write_route_weights(const char *dir, const ds4_weights *w, uint32_t start) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/meta.json", dir) >= (int)sizeof(path)) return false;
    FILE *fp = fopen(path, "w");
    if (!fp) return false;
    fprintf(fp, "{\"start\":%u,\"rows\":256,\"layers\":40,\"experts\":384,\"active\":6,\"head_bytes\":%llu,\"embedding_row_bytes\":%llu,\"weights\":[",
        start, (unsigned long long)(weight_bytes(w->output) + weight_bytes(w->output_norm)),
        (unsigned long long)(w->token_embd->bytes / DS4_N_VOCAB));
    for (unsigned il = 0; il < 40; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const ds4_tensor *always[] = {l->hc_attn_fn, l->hc_attn_scale, l->hc_attn_base,
            l->attn_norm, l->attn_q_a, l->attn_q_a_norm, l->attn_q_b, l->attn_kv,
            l->attn_kv_a_norm, l->attn_sinks, l->attn_output_a, l->attn_output_b,
            l->hc_ffn_fn, l->hc_ffn_scale, l->hc_ffn_base, l->ffn_norm,
            l->ffn_gate_inp, l->ffn_exp_probs_b, l->ffn_gate_shexp, l->ffn_up_shexp,
            l->ffn_down_shexp, l->engram_kv, l->engram_q_norm, l->engram_k_norm};
        uint64_t dense = 0;
        for (unsigned i = 0; i < sizeof(always) / sizeof(always[0]); i++) dense += weight_bytes(always[i]);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint64_t publish = ds41_kv_source(il) ? weight_bytes(l->attn_compressor_kv) +
            (ratio == 2 ? weight_bytes(l->attn_compressor_gate) : 0) : 0;
        const uint64_t boundary = ds41_kv_source(il) ? weight_bytes(l->attn_compressor_norm) +
            weight_bytes(l->indexer_attn_k) + weight_bytes(l->indexer_k_norm) : 0;
        const uint64_t query = ds41_index_source(il) ? weight_bytes(l->indexer_attn_q_b) +
            weight_bytes(l->indexer_proj) : 0;
        const uint64_t expert = (l->ffn_gate_exps->bytes + l->ffn_up_exps->bytes +
                                  l->ffn_down_exps->bytes) / DS4_N_EXPERT;
        fprintf(fp, "%s{\"layer\":%u,\"expert_bytes\":%llu,\"dense_always\":%llu,\"publish_every\":%llu,\"publish_boundary\":%llu,\"index_query\":%llu,\"ratio\":%u}",
            il ? "," : "", il, (unsigned long long)expert, (unsigned long long)dense,
            (unsigned long long)publish, (unsigned long long)boundary, (unsigned long long)query, ratio);
    }
    fprintf(fp, "]}\n");
    return fclose(fp) == 0;
}

static bool capture_tap(void *ud, uint32_t layer, uint32_t pos, uint32_t kind,
                         const ds4_gpu_tensor *src, uint64_t offset, uint64_t bytes) {
    tap_trace *t = ud;
    if (layer < 37 || layer > 39 || pos < t->start || pos - t->start >= t->count) return true;
    if (kind >= 4 || bytes != (kind ? 1u : 4u) * DS4_N_EMBD * sizeof(float)) return false;
    const uint32_t row = pos - t->start, tap = layer - 37u;
    if (!ds4_gpu_tensor_copy(t->value[kind], ((uint64_t)row * 3u + tap) * bytes, src, offset, bytes)) return false;
    t->seen[kind][tap][row] = 1;
    return true;
}

typedef struct {
    ds4_gpu_tensor *tensor[32];
    char name[32][64];
    uint64_t bytes[32];
    unsigned count;
} draft_trace;

static bool capture_trace(void *ud, const char *name, uint32_t stage,
                            const ds4_gpu_tensor *src, uint64_t bytes) {
    draft_trace *t = ud;
    if (t->count == 32) return false;
    const unsigned i = t->count++;
    snprintf(t->name[i], sizeof(t->name[i]), "%u-%s.f32", stage, name);
    t->bytes[i] = bytes;
    t->tensor[i] = ds4_gpu_tensor_alloc(bytes);
    return t->tensor[i] && ds4_gpu_tensor_copy(t->tensor[i], 0, src, 0, bytes);
}

static bool write_tensor(const char *dir, const char *name, const ds4_gpu_tensor *tensor, uint64_t bytes) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) return false;
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    void *data = malloc(bytes);
    const bool ok = data && ds4_gpu_tensor_read(tensor, 0, data, bytes) &&
        fwrite(data, 1, bytes, fp) == bytes;
    free(data);
    return fclose(fp) == 0 && ok;
}

static void free_trace(draft_trace *t) {
    for (unsigned i = 0; i < t->count; i++) ds4_gpu_tensor_free(t->tensor[i]);
    memset(t, 0, sizeof(*t));
}

static int check_live(const char *target_path, const char *support_path,
                        const char *prompt_path, bool capture) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_model support = {.fd = -1};
    ds4_dspark_weights dw = {0};
    ds41_dspark_graph draft = {0};
    draft_trace trace = {0};
    tap_trace taps = {0};
    route_trace routes = {0};
    const char *dump_root = getenv("DS4_DSPARK_DUMP_DIR");
    const char *tap_root = getenv("DS4_DSPARK_TAP_DUMP_DIR");
    const char *route_root = getenv("DS4_DSPARK_ROUTE_DUMP_DIR");
    ds4_gpu_tensor *last = NULL;
    ds4_tokens tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t prompt_bytes = 0;
    ds4_engine_options opt = {.model_path = target_path,
        .backend = DS4_BACKEND_METAL, .context_size = 32768, .power_percent = 100};
    CHECK(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    CHECK(prompt_bytes && ds4_engine_open(&engine, &opt) == 0);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &tokens);
    CHECK(tokens.len > 1 && tokens.len + 256 < 32768);
    CHECK(ds4_session_create(&session, engine, 32768) == 0);
    if (capture) {
        model_open(&support, support_path, true, false);
        CHECK(support_model_checkpoint_compatible(&support));
        const ds4_dspark_summary summary = model_dspark_summary(&support);
        dspark_weights_bind_optional(&dw, &support, &summary);
        CHECK(dw.v41 && !dw.missing_tensors && !dw.invalid_tensors && !dw.metadata_errors);
        CHECK(ds4_gpu_set_model_map_range(support.map, support.size, support.tensor_data_pos,
                                          support.size - support.tensor_data_pos, support.max_tensor_bytes));
        CHECK(ds41_dspark_capture_alloc(&session->ds41_graph, &dw));
        CHECK(ds41_dspark_alloc(&draft, &dw));
        if (route_root) {
            routes.start = (uint32_t)tokens.len;
            routes.value = ds4_gpu_tensor_alloc(256u * 40u * 6u * sizeof(int32_t));
            CHECK(routes.value);
            session->ds41_graph.dspark_capture->observe_routes = capture_routes;
            session->ds41_graph.dspark_capture->routes_ud = &routes;
        }
        if (tap_root) {
            CHECK(tokens.len >= 128);
            taps.start = (uint32_t)tokens.len - 128u;
            taps.count = 128u + 255u;
            for (unsigned kind = 0; kind < 4; kind++) {
                taps.value[kind] = ds4_gpu_tensor_alloc((uint64_t)taps.count * 3u *
                    (kind ? 1u : 4u) * DS4_N_EMBD * sizeof(float));
                CHECK(taps.value[kind]);
            }
            session->ds41_graph.dspark_capture->observe = capture_tap;
            session->ds41_graph.dspark_capture->observe_ud = &taps;
        }
    }
    CHECK(ds4_session_sync(session, &tokens, err, sizeof(err)) == 0);
    if (capture) CHECK(ds41_dspark_seed_capture(&draft, &support,
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
            char dump_dir[4096] = {0};
            const bool dump = dump_root && (i == 0 || i == 17 || i == 127);
            if (dump) {
                CHECK(snprintf(dump_dir, sizeof(dump_dir), "%s/%u", dump_root, i) < (int)sizeof(dump_dir));
                CHECK(mkdir(dump_dir, 0755) == 0);
                CHECK(ds41_dspark_capture_pack(draft.main_input, c, pos + 1u));
                CHECK(write_tensor(dump_dir, "hidden.f32", draft.main_input, 128u * bytes));
                draft.trace = capture_trace;
                draft.trace_ud = &trace;
            }
            const double t0 = now_sec();
            CHECK(ds41_dspark_forward(&draft, &engine->model, &engine->weights, &support,
                                        last, pos, token, proposals[i], confidence[i]));
            draft_ms += (now_sec() - t0) * 1000;
            if (dump) {
                draft.trace = NULL;
                for (unsigned j = 0; j < trace.count; j++)
                    CHECK(write_tensor(dump_dir, trace.name[j], trace.tensor[j], trace.bytes[j]));
                CHECK(write_tensor(dump_dir, "base_logits.f32", draft.base_logits,
                                    5u * DS4_N_VOCAB * sizeof(float)));
                CHECK(write_tensor(dump_dir, "confidence.f32", draft.confidence, 5u * sizeof(float)));
                char path[4096];
                CHECK(snprintf(path, sizeof(path), "%s/meta.json", dump_dir) < (int)sizeof(path));
                FILE *fp = fopen(path, "w");
                CHECK(fp);
                fprintf(fp, "{\"position\":%u,\"seed\":%d,\"proposals\":[%d,%d,%d,%d,%d]}\n",
                    pos, token, proposals[i][0], proposals[i][1], proposals[i][2], proposals[i][3], proposals[i][4]);
                CHECK(fclose(fp) == 0);
                free_trace(&trace);
            }
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
        if (route_root) {
            /* Capture the final emitted token's routes without emitting another token. */
            CHECK(ds4_session_eval(session, generated[255], err, sizeof(err)) == 0);
            for (unsigned row = 0; row < 256; row++)
                for (unsigned layer = 0; layer < 40; layer++) CHECK(routes.seen[row][layer]);
            CHECK(write_tensor(route_root, "routes.i32", routes.value, 256u * 40u * 6u * sizeof(int32_t)));
            CHECK(write_route_weights(route_root, &engine->weights, routes.start));
        }
        if (tap_root) {
            const char *names[] = {"raw_hc.f32", "mean.f32", "collapsed.f32", "normalized.f32"};
            for (unsigned kind = 0; kind < 4; kind++) {
                for (unsigned tap = 0; tap < 3; tap++)
                    for (unsigned row = 0; row < taps.count; row++) CHECK(taps.seen[kind][tap][row]);
                CHECK(write_tensor(tap_root, names[kind], taps.value[kind],
                    (uint64_t)taps.count * 3u * (kind ? 1u : 4u) * DS4_N_EMBD * sizeof(float)));
            }
            char path[4096];
            CHECK(snprintf(path, sizeof(path), "%s/meta.json", tap_root) < (int)sizeof(path));
            FILE *fp = fopen(path, "w");
            CHECK(fp);
            fprintf(fp, "{\"start\":%u,\"rows\":%u,\"first_generation_row\":127,\"layers\":[37,38,39],\"dim\":5120,\"hc\":4,\"generated\":[",
                    taps.start, taps.count);
            for (unsigned i = 0; i < 256; i++) fprintf(fp, "%s%d", i ? "," : "", generated[i]);
            fprintf(fp, "],\"pre_engram\":\"raw_hc.f32\",\"post_engram\":\"raw_hc.f32\",\"engram_on_tapped_layers\":false}\n");
            CHECK(fclose(fp) == 0);
        }
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
    free_trace(&trace);
    ds4_gpu_tensor_free(routes.value);
    for (unsigned i = 0; i < 4; i++) ds4_gpu_tensor_free(taps.value[i]);
    ds41_dspark_free(&draft);
    ds4_session_free(session);
    ds4_engine_close(engine);
    if (support.map) model_close(&support);
    ds4_tokens_free(&tokens);
    free(prompt);
    return rc;
}

static int replay_taps(char **argv) {
    int rc = 1;
    ds4_model target = {.fd = -1}, support = {.fd = -1};
    ds4_weights weights = {0};
    ds41_dspark_graph draft = {0};
    ds4_gpu_tensor *hidden = NULL, *last = NULL;
    float *features = NULL;
    FILE *fp = NULL;
    int32_t seeds[256];
    const uint32_t start = (uint32_t)strtoul(argv[6], NULL, 10);
    const size_t row_bytes = 3u * 5120u * sizeof(float);
    const size_t feature_bytes = 383u * row_bytes;
    CHECK(!strcmp(argv[7], "real") || !strcmp(argv[7], "zero") || !strcmp(argv[7], "constant"));
    g_ds4_shape = DS4_SHAPE_FLASH41;
    model_open(&target, argv[2], true, false);
    weights.token_embd = model_find_tensor(&target, "token_embd.weight");
    weights.output = model_find_tensor(&target, "output.weight");
    CHECK(weights.token_embd && weights.output && target.n_tensors == 2);
    CHECK(weights.token_embd->type == DS4_TENSOR_F16 && weights.output->type == DS4_TENSOR_Q8_0);
    model_open(&support, argv[3], true, false);
    CHECK(support_model_checkpoint_compatible(&support));
    const ds4_dspark_summary summary = model_dspark_summary(&support);
    ds4_dspark_weights dw;
    dspark_weights_bind_optional(&dw, &support, &summary);
    CHECK(dw.v41 && !dw.missing_tensors && !dw.invalid_tensors && !dw.metadata_errors);
    CHECK(dw.sliding_window == 128 && dw.block_size == 5 && dw.target_layer_count == 3);
    features = malloc(feature_bytes);
    CHECK(features);
    fp = fopen(argv[4], "rb");
    CHECK(fp && fread(features, 1, feature_bytes, fp) == feature_bytes && fgetc(fp) == EOF);
    fclose(fp); fp = NULL;
    fp = fopen(argv[5], "rb");
    CHECK(fp && fread(seeds, 1, sizeof(seeds), fp) == sizeof(seeds) && fgetc(fp) == EOF);
    fclose(fp); fp = NULL;
    if (strcmp(argv[7], "real"))
        for (size_t i = 0; i < feature_bytes / sizeof(float); i++)
            features[i] = !strcmp(argv[7], "zero") ? 0.0f : 1.0f;
    ds4_gpu_model_residency_skip(1);
    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map_range(target.map, target.size, target.tensor_data_pos,
        target.size - target.tensor_data_pos, target.max_tensor_bytes));
    CHECK(ds4_gpu_set_model_map_range(support.map, support.size, support.tensor_data_pos,
        support.size - support.tensor_data_pos, support.max_tensor_bytes));
    CHECK(ds41_dspark_alloc(&draft, &dw));
    hidden = ds4_gpu_tensor_alloc(feature_bytes);
    CHECK(hidden && ds4_gpu_tensor_write(hidden, 0, features, feature_bytes));
    CHECK(ds41_dspark_seed(&draft, &support, hidden, start, 128));
    for (uint32_t i = 0; i < 256; i++) {
        int32_t proposals[16] = {0};
        float confidence[16] = {0};
        last = ds4_gpu_tensor_view(hidden, (127u + i) * row_bytes, row_bytes);
        CHECK(last && ds41_dspark_forward(&draft, &target, &weights, &support,
            last, start + 127u + i, seeds[i], proposals, confidence));
        printf("{\"i\":%u,\"seed\":%d,\"proposals\":[%d,%d,%d,%d,%d],\"confidence\":[%.9g,%.9g,%.9g,%.9g,%.9g]}\n",
            i, seeds[i], proposals[0], proposals[1], proposals[2], proposals[3], proposals[4],
            confidence[0], confidence[1], confidence[2], confidence[3], confidence[4]);
        ds4_gpu_tensor_free(last); last = NULL;
    }
    rc = 0;
done:
    if (fp) fclose(fp);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(last);
    ds4_gpu_tensor_free(hidden);
    ds41_dspark_free(&draft);
    ds4_gpu_cleanup();
    model_close(&support);
    model_close(&target);
    free(features);
    return rc;
}

/* Two independent target states advance over the same greedy inputs. Every
 * batch row publishes logits; no rejected draft or rollback is involved. */
static int check_verify_rows(const char *target_path, const char *prompt_path,
                              uint32_t cap) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *batch = NULL;
    ds4_gpu_tensor *output = NULL;
    ds4_tokens prompt_tokens = {0};
    char *prompt = NULL, err[256] = {0};
    size_t prompt_bytes = 0;
    float *expected = NULL, *actual = NULL;
    ds4_engine_options opt = {.model_path = target_path,
        .backend = DS4_BACKEND_METAL, .context_size = 32768, .power_percent = 100};
    CHECK(cap >= 2 && cap <= 6);
    CHECK(imatrix_read_text_file(prompt_path, &prompt, &prompt_bytes));
    CHECK(prompt_bytes && ds4_engine_open(&engine, &opt) == 0);
    CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    ds4_encode_chat_prompt(engine, NULL, prompt, DS4_THINK_NONE, &prompt_tokens);
    CHECK(prompt_tokens.len > 1 && prompt_tokens.len + 256 < 32768);
    CHECK(ds4_session_create(&scalar, engine, 32768) == 0);
    CHECK(ds4_session_create(&batch, engine, 32768) == 0);
    CHECK(ds4_session_sync(scalar, &prompt_tokens, err, sizeof(err)) == 0);
    CHECK(ds4_session_sync(batch, &prompt_tokens, err, sizeof(err)) == 0);
    const size_t row_bytes = (size_t)DS4_N_VOCAB * sizeof(float);
    CHECK(!memcmp(scalar->logits, batch->logits, row_bytes));
    expected = malloc(cap * row_bytes);
    actual = malloc(cap * row_bytes);
    output = ds4_gpu_tensor_alloc(cap * row_bytes);
    CHECK(expected && actual && output);
    double scalar_ms = 0, batch_ms = 0;
    uint32_t mismatches = 0, different_rows = 0, batches = 0;
    int generated[256];
    for (uint32_t first = 0; first < 256;) {
        const uint32_t rows = 256u - first < cap ? 256u - first : cap;
        ds41_gpu_graph *graphs[6];
        for (uint32_t i = 0; i < rows; i++) {
            graphs[i] = &batch->ds41_graph;
            const int token = sample_argmax(scalar->logits, DS4_N_VOCAB);
            CHECK(!vocab_token_is_generation_stop(&engine->vocab, token));
            generated[first + i] = token;
            const double t0 = now_sec();
            CHECK(ds41_graph_step(&scalar->ds41_graph, &engine->model,
                                   &engine->weights, token, scalar->logits));
            scalar_ms += (now_sec() - t0) * 1000;
            memcpy(expected + (size_t)i * DS4_N_VOCAB, scalar->logits, row_bytes);
        }
        const double t0 = now_sec();
        if (rows == 1) {
            CHECK(ds41_graph_step(&batch->ds41_graph, &engine->model,
                                   &engine->weights, generated[first], actual));
        } else {
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds41_graph_step_batch_outputs(graphs, generated + first, (int)rows,
                rows, &engine->model, &engine->weights, output));
            CHECK(ds4_gpu_end_commands());
            CHECK(ds4_gpu_tensor_read(output, 0, actual, rows * row_bytes));
        }
        const double elapsed = (now_sec() - t0) * 1000;
        batch_ms += elapsed;
        CHECK(batch->ds41_graph.pos == scalar->ds41_graph.pos);
        CHECK(!memcmp(&batch->ds41_graph.history, &scalar->ds41_graph.history,
                       sizeof(batch->ds41_graph.history)));
        for (uint32_t i = 0; i < rows; i++) {
            const float *a = expected + (size_t)i * DS4_N_VOCAB;
            const float *b = actual + (size_t)i * DS4_N_VOCAB;
            const int want = sample_argmax(a, DS4_N_VOCAB);
            const int got = sample_argmax(b, DS4_N_VOCAB);
            bool finite = true;
            for (uint32_t j = 0; j < DS4_N_VOCAB; j++) finite &= isfinite(b[j]);
            CHECK(finite);
            const bool equal = !memcmp(a, b, row_bytes);
            different_rows += !equal;
            mismatches += want != got;
            fprintf(stderr, "verify row=%u rows=%u expected=%d actual=%d logits_exact=%d\n",
                    first + i, rows, want, got, equal);
        }
        fprintf(stderr, "verify_batch first=%u rows=%u ms=%.6f\n", first, rows, elapsed);
        batches++;
        first += rows;
    }
    fprintf(stderr, "V4.1 verify prompt_tokens=%d cap=%u rows=256 batches=%u scalar_ms=%.3f batch_ms=%.3f top1_mismatches=%u different_logit_rows=%u\n",
            prompt_tokens.len, cap, batches, scalar_ms, batch_ms, mismatches, different_rows);
    fprintf(stderr, "target_tokens=");
    for (uint32_t i = 0; i < 256; i++) {
        fprintf(stderr, "%s%d", i ? "," : "", generated[i]);
        size_t len = 0;
        char *piece = ds4_token_text(engine, generated[i], &len);
        CHECK(piece);
        fwrite(piece, 1, len, stdout);
        free(piece);
    }
    fputc('\n', stderr);
    rc = mismatches ? 1 : 0;
done:
    if (rc) fprintf(stderr, "V4.1 verify probe failed: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(output);
    free(expected); free(actual); free(prompt);
    ds4_tokens_free(&prompt_tokens);
    ds4_session_free(batch); ds4_session_free(scalar);
    ds4_engine_close(engine);
    return rc;
}

static bool numerical_logits(const float *reference, const float *candidate,
                              unsigned step, unsigned session, const char *comparison) {
    const int want = sample_argmax(reference, DS4_N_VOCAB);
    const int got = sample_argmax(candidate, DS4_N_VOCAB);
    double sum = 0, sum2 = 0, maximum = 0;
    float runner_up = -INFINITY, candidate_runner_up = -INFINITY;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if (!isfinite(reference[i]) || !isfinite(candidate[i])) return false;
        const double error = fabs((double)reference[i] - candidate[i]);
        maximum = fmax(maximum, error);
        sum += error;
        sum2 += error * error;
        if ((int)i != want) runner_up = fmaxf(runner_up, reference[i]);
        if ((int)i != got) candidate_runner_up = fmaxf(candidate_runner_up, candidate[i]);
    }
    printf("LOGITS step=%u session=%u comparison=%s max_abs=%.9g mean_abs=%.9g rms=%.9g expected=%d actual=%d margin=%.9g candidate_margin=%.9g winner_gap=%.9g exact=%d\n",
           step, session, comparison, maximum, sum / DS4_N_VOCAB,
           sqrt(sum2 / DS4_N_VOCAB), want, got,
           (double)reference[want] - runner_up,
           (double)candidate[got] - candidate_runner_up,
           (double)reference[want] - reference[got],
           !memcmp(reference, candidate, (size_t)DS4_N_VOCAB * sizeof(float)));
    return true;
}

/* Feed every path the scalar continuation, including after a disagreement.
 * The binary stream permits a byte comparison across independent repeats. */
static int numerical_sessions(char **argv) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *scalar[3] = {0}, *batch[2][6] = {{0}};
    ds4_decode_item items[6] = {0};
    ds4_tokens prompts[3] = {{0}};
    char *text[3] = {0}, err[256] = {0};
    FILE *raw = NULL;
    const ds4_engine_options opt = {.model_path = argv[2],
        .backend = DS4_BACKEND_METAL, .context_size = 32768, .power_percent = 100};
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    CHECK(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    CHECK(!getenv("DS4_METAL_DISABLE_V41_EXPERT_UNION_GATE"));
    CHECK(setenv("DS4_METAL_DISABLE_V41_EXPERT_PAIR_REUSE", "1", 1) == 0);
    for (int w = 0; w < 3; w++) {
        size_t bytes = 0;
        CHECK(imatrix_read_text_file(argv[3+w], &text[w], &bytes) && bytes);
        ds4_encode_chat_prompt(engine, NULL, text[w], DS4_THINK_NONE, &prompts[w]);
        CHECK(prompts[w].len > 1 && prompts[w].len + 256 < 32768);
        CHECK(ds4_session_create(&scalar[w], engine, 32768) == 0);
        CHECK(ds4_session_sync(scalar[w], &prompts[w], err, sizeof(err)) == 0);
        fprintf(stderr, "NUMERICAL_PREFILL scalar=%d tokens=%d\n", w, prompts[w].len);
    }
    const size_t row_bytes = (size_t)DS4_N_VOCAB * sizeof(float);
    for (int variant = 0; variant < 2; variant++) {
        for (int i = 0; i < 6; i++) {
            CHECK(ds4_session_create(&batch[variant][i], engine, 32768) == 0);
            CHECK(ds4_session_sync(batch[variant][i], &prompts[i%3], err, sizeof(err)) == 0);
            CHECK(!memcmp(scalar[i%3]->logits, batch[variant][i]->logits, row_bytes));
            fprintf(stderr, "NUMERICAL_PREFILL variant=%d session=%d\n", variant, i);
        }
    }
    raw = fopen(argv[6], "wb");
    CHECK(raw);
    for (unsigned step = 0; step < 256; step++) {
        int tokens[3];
        for (int w = 0; w < 3; w++) {
            tokens[w] = sample_argmax(scalar[w]->logits, DS4_N_VOCAB);
            CHECK(!vocab_token_is_generation_stop(&engine->vocab, tokens[w]));
            ds4_decode_item item = {.session = scalar[w], .token = tokens[w]};
            CHECK(ds4_sessions_eval_batch(&item, 1, err, sizeof(err)) == 0);
            CHECK(fwrite(scalar[w]->logits, 1, row_bytes, raw) == row_bytes);
            printf("TOKEN step=%u workload=%d token=%d\n", step, w, tokens[w]);
        }
        for (int variant = 0; variant < 2; variant++) {
            if (variant) CHECK(unsetenv("DS4_METAL_DISABLE_V41_EXPERT_PAIR_REUSE") == 0);
            else CHECK(setenv("DS4_METAL_DISABLE_V41_EXPERT_PAIR_REUSE", "1", 1) == 0);
            for (int i = 0; i < 6; i++) {
                items[i].session = batch[variant][i];
                items[i].token = tokens[i%3];
            }
            CHECK(ds41_sessions_batch_supported(items, 6, engine));
            CHECK(ds4_sessions_eval_batch(items, 6, err, sizeof(err)) == 0);
            for (int i = 0; i < 6; i++) {
                CHECK(batch[variant][i]->ds41_graph.pos == scalar[i%3]->ds41_graph.pos);
                CHECK(!memcmp(&batch[variant][i]->ds41_graph.history,
                              &scalar[i%3]->ds41_graph.history,
                              sizeof(scalar[i%3]->ds41_graph.history)));
                CHECK(fwrite(batch[variant][i]->logits, 1, row_bytes, raw) == row_bytes);
                CHECK(numerical_logits(scalar[i%3]->logits, batch[variant][i]->logits,
                      step, (unsigned)i, variant ? "paired_scalar" : "grouped_scalar"));
                if (variant) CHECK(numerical_logits(batch[0][i]->logits, batch[1][i]->logits,
                                                    step, (unsigned)i, "paired_grouped"));
            }
        }
        if (step % 16 == 0) fprintf(stderr, "NUMERICAL_STEP step=%u\n", step);
    }
    CHECK(fflush(raw) == 0);
    fprintf(stderr, "NUMERICAL_COMPLETE positions=256 sessions=6 workloads=3 vocab=%u\n", DS4_N_VOCAB);
    rc = 0;
done:
    if (rc) fprintf(stderr, "numerical session audit failed: %s\n", err);
    if (raw && fclose(raw)) rc = 1;
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    for (int variant = 0; variant < 2; variant++)
        for (int i = 0; i < 6; i++) ds4_session_free(batch[variant][i]);
    for (int w = 0; w < 3; w++) {
        ds4_session_free(scalar[w]); ds4_tokens_free(&prompts[w]); free(text[w]);
    }
    ds4_engine_close(engine);
    return rc;
}

/* Exercise the public serving API with independent session state. */
static int bench_sessions(const char *target_path, const char *code_path,
                           const char *english_path, int count) {
    int rc = 1;
    ds4_engine *engine = NULL;
    ds4_session *sessions[6] = {0};
    ds4_decode_item items[6] = {0};
    ds4_tokens prompts[2] = {{0}};
    char *text[2] = {0}, err[256] = {0};
    size_t bytes[2];
    int emitted[6][256];
    ds4_engine_options opt = {.model_path = target_path,
        .backend = DS4_BACKEND_METAL, .context_size = 32768, .power_percent = 100};
    CHECK(count == 1 || count == 2 || count == 4 || count == 6);
    CHECK(imatrix_read_text_file(code_path, &text[0], &bytes[0]));
    CHECK(imatrix_read_text_file(english_path, &text[1], &bytes[1]));
    CHECK(ds4_engine_open(&engine, &opt) == 0);
    for (int i = 0; i < 2; i++)
        ds4_encode_chat_prompt(engine, NULL, text[i], DS4_THINK_NONE, &prompts[i]);
    for (int i = 0; i < count; i++) {
        CHECK(ds4_session_create(&sessions[i], engine, 32768) == 0);
        CHECK(ds4_session_sync(sessions[i], &prompts[i % 2], err, sizeof(err)) == 0);
        items[i].session = sessions[i];
    }
    if (count > 1) CHECK(ds41_sessions_batch_supported(items, count, engine));
    fprintf(stderr, "SESSION_PATH count=%d native_ds41=%d prompts=%d,%d\n",
            count, count > 1, prompts[0].len, prompts[1].len);
    double total_ms = 0;
    for (int step = 0; step < 256; step++) {
        for (int i = 0; i < count; i++) {
            items[i].token = sample_argmax(sessions[i]->logits, DS4_N_VOCAB);
            CHECK(!vocab_token_is_generation_stop(&engine->vocab, items[i].token));
            emitted[i][step] = items[i].token;
        }
        const double t0 = now_sec();
        CHECK(ds4_sessions_eval_batch(items, count, err, sizeof(err)) == 0);
        const double ms = (now_sec() - t0) * 1000;
        if (step >= 16) total_ms += ms;
        fprintf(stderr, "SESSION_STEP step=%d count=%d ms=%.6f\n", step, count, ms);
        if (metal_graph_gpu_stage_timestamps())
            ds4_gpu_stage_report("sessions", (uint32_t)step, (uint32_t)count);
    }
    for (int i = 0; i < count; i++) {
        printf("session=%d tokens=", i);
        for (int j = 0; j < 256; j++) printf("%s%d", j ? "," : "", emitted[i][j]);
        putchar('\n');
    }
    fprintf(stderr, "SESSION_RESULT count=%d measured_steps=240 mean_ms=%.6f aggregate_tps=%.6f\n",
            count, total_ms / 240, count * 240000.0 / total_ms);
    rc = 0;
done:
    if (rc) fprintf(stderr, "session benchmark failed: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    for (int i = 0; i < 6; i++) ds4_session_free(sessions[i]);
    for (int i = 0; i < 2; i++) { ds4_tokens_free(&prompts[i]); free(text[i]); }
    ds4_engine_close(engine);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 7 && !strcmp(argv[1], "--sessions-numerical"))
        return numerical_sessions(argv);
    if (argc == 6 && !strcmp(argv[1], "--sessions"))
        return bench_sessions(argv[2], argv[3], argv[4], atoi(argv[5]));
    if (argc == 5 && !strcmp(argv[1], "--verify-rows"))
        return check_verify_rows(argv[2], argv[3], (uint32_t)atoi(argv[4]));
    if (argc == 8 && !strcmp(argv[1], "--replay")) return replay_taps(argv);
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
