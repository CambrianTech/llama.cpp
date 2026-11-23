// Continuum fork: the routes our server adds (page IO, learning, adapters, cells, train control), kept
// in one place and reached by one named hook in each of server.cpp's route functions, so upstream's
// server.cpp stays upstream's.
#pragma once

#include "server-context.h"
#include "server-http.h"

// Registers our routes beside upstream's (server_register_routes calls this once).
template <typename Wrap>
static void continuum_register_routes(server_http_context & ctx_http, server_routes & routes, Wrap wrap) {
    ctx_http.post("/v1/chat/completions/cells",   wrap(routes.post_chat_completions_cells));
    ctx_http.post("/lora-adapters/load",          wrap(routes.post_lora_adapters_load));
    ctx_http.post("/lora-adapters/fresh",         wrap(routes.post_lora_adapters_fresh));
    ctx_http.post("/lora-adapters/save",          wrap(routes.post_lora_adapters_save));
    ctx_http.post("/lora-adapters/unload",        wrap(routes.post_lora_adapters_unload));
    ctx_http.post("/learn",                       wrap(routes.post_learn));
    ctx_http.get ("/learn/grants",                wrap(routes.get_learn_grants));
    ctx_http.post("/learn/residency",             wrap(routes.post_learn_residency));
    ctx_http.post("/learn/residency/release",     wrap(routes.release_learn_residency));
    ctx_http.post("/learn/residency/revoke",      wrap(routes.revoke_learn_residency));
    ctx_http.post("/learn/preflight",             wrap(routes.post_learn_preflight));
    ctx_http.post("/learn/grants",                wrap(routes.post_learn_grants));
    ctx_http.get ("/learn/events",                wrap(routes.get_learn_events));
    ctx_http.post("/prompt-cache/limit",          wrap(routes.post_prompt_cache_limit));
    ctx_http.get ("/page-io",                     wrap(routes.get_page_io));
    ctx_http.post("/page-io/budget",              wrap(routes.post_page_io_budget));
    ctx_http.post("/learn/grants/revoke",         wrap(routes.revoke_learn_grant));
    ctx_http.get ("/train",                       wrap(routes.get_train));
    ctx_http.post("/train",                       wrap(routes.post_train));
    ctx_http.post("/train/cancel",                wrap(routes.post_train_cancel));
    ctx_http.post("/train/measure",               wrap(routes.post_train_measure));
    ctx_http.post("/train/pause",                 wrap(routes.post_train_pause));
    ctx_http.post("/train/resume",                wrap(routes.post_train_resume));
    ctx_http.get ("/cells",                       wrap(routes.get_cells));
}

// A router forwards our routes to its instances like upstream's (the router section calls this once).
template <typename Proxies>
static void continuum_route_proxies(server_routes & routes, Proxies & proxies) {
    routes.post_chat_completions_cells = proxies.proxy_post;
    routes.post_lora_adapters_load     = proxies.proxy_post;
    routes.post_lora_adapters_fresh    = proxies.proxy_post;
    routes.post_lora_adapters_save     = proxies.proxy_post;
    routes.post_lora_adapters_unload   = proxies.proxy_post;
    routes.post_learn                  = proxies.proxy_post;
    routes.get_learn_grants            = proxies.proxy_get;
    routes.get_learn_events            = proxies.proxy_get;
    routes.post_learn_grants           = proxies.proxy_post;
    routes.post_learn_preflight        = proxies.proxy_post;
    routes.post_prompt_cache_limit     = proxies.proxy_post;
    routes.get_page_io                 = proxies.proxy_get;
    routes.post_page_io_budget         = proxies.proxy_post;
    routes.post_learn_residency        = proxies.proxy_post;
    routes.release_learn_residency     = proxies.proxy_post;
    routes.revoke_learn_residency      = proxies.proxy_post;
    routes.revoke_learn_grant          = proxies.proxy_post;
    routes.get_cells                   = proxies.proxy_get;
}
