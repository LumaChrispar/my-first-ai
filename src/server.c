/* server.c - a small, bounded, loopback-only HTTP server.
 *
 * Security posture, stated plainly:
 *   - the socket binds to 127.0.0.1 by default and a non-loopback host is
 *     refused outright;
 *   - request headers and bodies are length-limited, and Content-Length is
 *     checked before anything is allocated;
 *   - the UI is served as text and the browser renders all messages with
 *     textContent, so message text can never become executable HTML;
 *   - errors name the operational problem only. Stack traces and local paths
 *     are never sent to the browser;
 *   - prompts are not written to any log.
 */
#include "server.h"
#include "jsonstr.h"
#include "util.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
typedef SOCKET socket_t;
#define CLOSESOCK closesocket
#define ASTER_WINSOCK 1
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR   (-1)
#define CLOSESOCK close
#endif

#define MAX_HEADER_BYTES 8192
#define MAX_BODY_BYTES   32768
#define MAX_PROMPT_BYTES 4096
#define READ_CHUNK       4096
#define RECV_TIMEOUT_SEC 15

/* ---------------------------------------------------------------- sockets */

static void send_all(socket_t c, const char *data, size_t len) {
    while (len > 0) {
        int n = send(c, data, (int)(len > 0x40000000u ? 0x40000000u : len), 0);
        if (n <= 0) return;
        data += n;
        len -= (size_t)n;
    }
}

static void respond(socket_t c, int code, const char *reason, const char *type,
                    const char *extra, const char *body) {
    char head[768];
    size_t blen = body ? strlen(body) : 0;
    int n = snprintf(head, sizeof head,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "Cache-Control: no-store\r\n"
        "%s"
        "\r\n",
        code, reason, type, blen, extra ? extra : "");
    if (n < 0) return;
    send_all(c, head, (size_t)n);
    if (blen) send_all(c, body, blen);
}

static void respond_json(socket_t c, int code, const char *reason, const char *json) {
    respond(c, code, reason, "application/json; charset=utf-8", NULL, json);
}

static void respond_error(socket_t c, int code, const char *reason, const char *code_str,
                          const char *message) {
    char esc[1024];
    if (json_escape(message, esc, sizeof esc) != 0) strcpy(esc, "request failed");
    char body[1200];
    int n = snprintf(body, sizeof body, "{\"error\":\"%s\",\"message\":\"%s\"}", code_str, esc);
    respond_json(c, code, reason, (n > 0 && (size_t)n < sizeof body) ? body : "{\"error\":\"internal\"}");
}

/* ---------------------------------------------------------------- helpers */

/* memmem() is a GNU extension that MinGW does not provide, so search here. */
static const char *find_bytes(const char *hay, size_t hlen, const char *needle, size_t nlen) {
    if (nlen == 0 || hlen < nlen) return NULL;
    for (size_t i = 0; i + nlen <= hlen; ++i)
        if (hay[i] == needle[0] && memcmp(hay + i, needle, nlen) == 0) return hay + i;
    return NULL;
}

static int ci_equal_n(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int ca = tolower((unsigned char)a[i]);
        int cb = tolower((unsigned char)b[i]);
        if (ca != cb) return 0;
    }
    return 1;
}

static int is_loopback(const char *host) {
    struct in_addr a4;
    struct in6_addr a6;
    if (inet_pton(AF_INET, host, &a4) == 1) {
        unsigned char b0 = (unsigned char)(ntohl(a4.s_addr) >> 24);
        return b0 == 127;
    }
    if (inet_pton(AF_INET6, host, &a6) == 1) {
        static const unsigned char loop[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
        return memcmp(&a6, loop, 16) == 0;
    }
    return 0;
}

/* Case-insensitive header lookup inside the request head. */
static int header_value(const char *head, size_t len, const char *name,
                        char *out, size_t cap) {
    size_t nlen = strlen(name);
    const char *p = head;
    const char *end = head + len;
    const char *first = strchr(head, '\n');
    if (!first) return 0;
    p = first + 1;
    while (p < end) {
        const char *eol = (const char *)memchr(p, '\n', (size_t)(end - p));
        size_t linelen = eol ? (size_t)(eol - p) : (size_t)(end - p);
        if (linelen && p[linelen - 1] == '\r') --linelen;
        if (linelen == 0) break;                       /* end of headers */
        if (linelen > nlen && p[nlen] == ':' && ci_equal_n(p, name, nlen)) {
            size_t v = nlen + 1;
            while (v < linelen && (p[v] == ' ' || p[v] == '\t')) ++v;
            size_t outn = linelen - v;
            if (outn >= cap) outn = cap - 1;
            memcpy(out, p + v, outn);
            out[outn] = '\0';
            return 1;
        }
        if (!eol) break;
        p = eol + 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ routes */

static void serve_index(AsterServer *s, socket_t c) {
    (void)s;
    size_t len = 0; int ok = 0;
    char *html = aster_read_file("index.html", &len, &ok);
    if (!html) {
        respond(c, 500, "Internal Server Error", "text/plain; charset=utf-8", NULL,
                "index.html could not be read. Start the server from the project folder.");
        return;
    }
    respond(c, 200, "OK", "text/html; charset=utf-8", NULL, html);
    free(html);
}

static void route_status(AsterServer *s, socket_t c) {
    char body[1400];
    if (!s->model) {
        snprintf(body, sizeof body,
            "{\"ready\":false,\"status\":\"model_not_trained\","
            "\"detail\":\"No checkpoint is loaded, so no answer can be generated.\","
            "\"how_to_train\":\"aster train --data data/demo_chat.jsonl "
            "--validation data/demo_valid.jsonl --out models/aster-small.bin --seed 1234\","
            "\"network\":\"none\",\"tools\":\"none\"}");
        respond_json(c, 200, "OK", body);
        return;
    }
    const AsterConfig *g = &s->model->cfg;
    int n = snprintf(body, sizeof body,
        "{\"ready\":true,\"status\":\"ready\","
        "\"model\":\"%s\",\"train_step\":%u,\"seed\":%u,"
        "\"architecture\":{\"n_layer\":%d,\"n_head\":%d,\"d_model\":%d,"
        "\"d_ff\":%d,\"context_tokens\":%d,\"vocab\":%d},"
        "\"parameters\":%d,\"tokenizer\":\"byte-v1\","
        "\"max_new_tokens\":%d,\"temperature\":%.3f,\"top_k\":%d,"
        "\"prompt_budget_bytes\":%d,"
        "\"network\":\"none\",\"tools\":\"none\",\"persistent_memory\":\"none\"}",
        s->name, s->step, s->seed,
        g->n_layer, g->n_head, g->d_model, g->d_ff, g->context, g->vocab,
        s->model->off.total, s->max_new_tokens, (double)s->temperature, s->top_k,
        aster_prompt_budget(g, s->max_new_tokens, (int)strlen(s->system ? s->system : "")));
    respond_json(c, 200, "OK", (n > 0 && (size_t)n < sizeof body) ? body
                                                                    : "{\"ready\":false}");
}

static void route_chat(AsterServer *s, socket_t c, const char *body, size_t blen) {
    if (!s->model) {
        respond_error(c, 503, "Service Unavailable", "model_not_trained",
                      "No model checkpoint is loaded, so there is nothing to generate from. "
                      "Train one with: aster train --data data/demo_chat.jsonl "
                      "--out models/aster-small.bin --seed 1234");
        return;
    }

    char message[8192];
    char jerr[160] = {0};
    int rc = json_get_string(body, blen, "message", message, sizeof message, jerr, sizeof jerr);
    if (rc == -1) { respond_error(c, 400, "Bad Request", "invalid_json", jerr); return; }
    if (rc == -2) { respond_error(c, 400, "Bad Request", "missing_field",
                                  "The request needs a \"message\" string."); return; }
    if (rc == -3) { respond_error(c, 413, "Payload Too Large", "message_too_long",
                                  "That message is longer than this server accepts."); return; }
    if (!message[0]) { respond_error(c, 400, "Bad Request", "empty_message",
                                     "The message was empty."); return; }

    const AsterConfig *g = &s->model->cfg;
    int budget = aster_prompt_budget(g, s->max_new_tokens, (int)strlen(s->system ? s->system : ""));
    if (budget < 1) {
        respond_error(c, 500, "Internal Server Error", "no_context_budget",
                      "This model's context is too small to hold a prompt and a reply.");
        return;
    }
    if (strlen(message) > (size_t)MAX_PROMPT_BYTES) {
        respond_error(c, 413, "Payload Too Large", "message_too_long",
                      "That message is longer than this server accepts.");
        return;
    }
    int truncated_prompt = 0;
    size_t msglen = strlen(message);
    if ((int)msglen > budget) {
        /* The byte tokenizer spends one token per byte, so the budget is also
         * a byte count. Keep the END of the message, which is the part the
         * user just added, and start on a UTF-8 lead byte. */
        size_t keep = (size_t)budget;
        size_t start = msglen - keep;
        while (start < msglen && ((unsigned char)message[start] & 0xC0) == 0x80) ++start;
        memmove(message, message + start, msglen - start + 1);
        truncated_prompt = 1;
    }
    if (!message[0]) {
        respond_error(c, 400, "Bad Request", "prompt_too_long_for_context",
                      "This model has a very small context, and this message does not fit "
                      "in it. Try a much shorter message.");
        return;
    }

    GenParams gp;
    gp.max_new_tokens = s->max_new_tokens;
    gp.temperature = s->temperature;
    gp.top_k = s->top_k;
    gp.seed = s->seed;
    gp.system = s->system;

    char *reply = (char *)aster_xmalloc(MAX_BODY_BYTES);
    int hit_end = 0, truncated_ctx = 0;
    size_t rn = aster_generate(s->model, message, &gp, reply, MAX_BODY_BYTES, &hit_end, &truncated_ctx);

    if (rn == 0 && !hit_end) {
        free(reply);
        respond_error(c, 500, "Internal Server Error", "empty_generation",
                      "The model produced no output for this prompt.");
        return;
    }

    /* A control byte can expand sixfold, so the escape buffer is generous. */
    char *esc_reply = (char *)aster_xmalloc(MAX_BODY_BYTES * 6 + 16);
    if (json_escape(reply, esc_reply, MAX_BODY_BYTES * 6 + 16) != 0) {
        free(reply); free(esc_reply);
        respond_error(c, 500, "Internal Server Error", "reply_too_long",
                      "The generated reply was too long to send.");
        return;
    }
    free(reply);

    char note[256] = {0};
    if (truncated_prompt)
        snprintf(note, sizeof note, "Only the end of your message fit in this model's "
                 "%d-token context, so older text was dropped.", g->context);
    else if (truncated_ctx)
        snprintf(note, sizeof note, "Generation stopped because the %d-token context filled up.", g->context);
    char esc_note[1024];
    if (json_escape(note, esc_note, sizeof esc_note) != 0) esc_note[0] = '\0';

    size_t out_cap = strlen(esc_reply) + strlen(esc_note) + strlen(s->name) + 256;
    char *out = (char *)aster_xmalloc(out_cap);
    snprintf(out, out_cap,
        "{\"reply\":\"%s\",\"note\":\"%s\",\"hit_end_marker\":%s,"
        "\"model\":\"%s\",\"train_step\":%u,\"max_new_tokens\":%d}",
        esc_reply, esc_note, hit_end ? "true" : "false", s->name, s->step, s->max_new_tokens);
    respond_json(c, 200, "OK", out);
    free(out);
    free(esc_reply);
}

/* Reads the request line + headers, then exactly Content-Length body bytes. */
static void handle_client(AsterServer *s, socket_t c) {
    char head[MAX_HEADER_BYTES + 1];
    size_t got = 0;

    while (got < MAX_HEADER_BYTES) {
        int n = recv(c, head + got, (int)(MAX_HEADER_BYTES - got), 0);
        if (n <= 0) break;
        got += (size_t)n;
        if (find_bytes(head, got, "\r\n\r\n", 4) || find_bytes(head, got, "\n\n", 2)) break;
    }
    if (got == 0) return;
    head[got] = '\0';

    const char *end = find_bytes(head, got, "\r\n\r\n", 4);
    size_t head_len;
    if (end) head_len = (size_t)(end - head) + 4;
    else {
        end = find_bytes(head, got, "\n\n", 2);
        if (!end) {
            respond_error(c, 431, "Request Header Fields Too Large", "headers_too_large",
                          "The request headers were incomplete or too large.");
            return;
        }
        head_len = (size_t)(end - head) + 2;
    }

    char method[16] = {0}, path[512] = {0}, clen[32] = {0};
    if (sscanf(head, "%15s %511s", method, path) != 2) {
        respond_error(c, 400, "Bad Request", "bad_request_line", "The request line could not be read.");
        return;
    }
    /* Ignore any query string; the API takes everything in the body. */
    char *q = strchr(path, '?');
    if (q) *q = '\0';

    size_t have = got - head_len;
    char *body = (char *)aster_xcalloc(1, MAX_BODY_BYTES + 1);
    if (have > MAX_BODY_BYTES) have = MAX_BODY_BYTES;
    if (have) memcpy(body, head + head_len, have);

    int is_post = (strcmp(method, "POST") == 0);
    if (is_post) {
        long want = 0;
        if (header_value(head, head_len, "Content-Length", clen, sizeof clen)) {
            char *endp = NULL;
            want = strtol(clen, &endp, 10);
            if (!endp || *endp != '\0' || want < 0) {
                respond_error(c, 400, "Bad Request", "bad_content_length",
                              "The Content-Length header was not a valid number.");
                free(body);
                return;
            }
        } else {
            respond_error(c, 411, "Length Required", "length_required",
                          "This server needs a Content-Length header.");
            free(body);
            return;
        }
        if (want > MAX_BODY_BYTES) {
            respond_error(c, 413, "Payload Too Large", "body_too_large",
                          "The request body is larger than this server accepts.");
            free(body);
            return;
        }
        /* Drain the rest, one bounded read at a time. */
        size_t total = have;
        while (total < (size_t)want) {
            size_t room = (size_t)want - total;
            if (room > READ_CHUNK) room = READ_CHUNK;
            int n = recv(c, body + total, (int)room, 0);
            if (n <= 0) break;
            total += (size_t)n;
        }
        if (total < (size_t)want) {
            respond_error(c, 400, "Bad Request", "truncated_body",
                          "The request body ended before Content-Length was reached.");
            free(body);
            return;
        }
        body[total] = '\0';
        if (strcmp(path, "/api/chat") == 0) {
            route_chat(s, c, body, total);
        } else {
            respond_error(c, 404, "Not Found", "not_found", "No such endpoint.");
        }
        free(body);
        return;
    }

    free(body);
    if (strcmp(method, "GET") != 0) {
        respond_error(c, 405, "Method Not Allowed", "method_not_allowed", "Only GET and POST are supported.");
        return;
    }
    if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) { serve_index(s, c); return; }
    if (strcmp(path, "/api/status") == 0 || strcmp(path, "/api/health") == 0) { route_status(s, c); return; }
    if (strcmp(path, "/favicon.ico") == 0) { respond(c, 204, "No Content", "image/x-icon", NULL, ""); return; }
    respond_error(c, 404, "Not Found", "not_found", "No such endpoint.");
}

/* ------------------------------------------------------------------- serve */

int aster_serve(AsterServer *s, const char *host, int port) {
    if (!is_loopback(host)) {
        aster_fail("refusing to bind to %s. This server only listens on loopback "
                   "addresses such as 127.0.0.1; Aster never exposes a model to the network.",
                   host);
    }
    if (port < 1 || port > 65535) aster_fail("port %d is out of range", port);

#ifdef ASTER_WINSOCK
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) aster_fail("Winsock startup failed");
#endif

    socket_t srv = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (srv == INVALID_SOCKET) aster_fail("could not create a socket");
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof yes);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        CLOSESOCK(srv);
        aster_fail("could not parse the host address");
    }
    if (bind(srv, (struct sockaddr *)&addr, sizeof addr) == SOCKET_ERROR) {
        CLOSESOCK(srv);
        aster_fail("could not bind %s:%d. Another copy of Aster may already be running.",
                   host, port);
    }
    if (listen(srv, 16) == SOCKET_ERROR) {
        CLOSESOCK(srv);
        aster_fail("could not listen on %s:%d", host, port);
    }

    if (s->model) {
        aster_info("model:    %s (%d parameters, %d-token context)",
                   s->name, s->model->off.total, s->model->cfg.context);
        aster_info("step:     %u, seed %u, max %d new tokens",
                   s->step, s->seed, s->max_new_tokens);
    } else {
        aster_warn("no checkpoint is loaded. The chat route will answer 503 until you "
                   "train a model: aster train --data data/demo_chat.jsonl "
                   "--out models/aster-small.bin --seed 1234");
    }
    aster_info("listening on http://%s:%d  (loopback only, no network access)", host, port);
    aster_info("open that address in a browser. Press Ctrl+C to stop.");

    for (;;) {
        socket_t c = accept(srv, NULL, NULL);
        if (c == INVALID_SOCKET) continue;
#ifdef ASTER_WINSOCK
        DWORD tv = RECV_TIMEOUT_SEC * 1000;
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof tv);
        DWORD tv2 = RECV_TIMEOUT_SEC * 1000;
        setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv2, sizeof tv2);
#else
        struct timeval tv = { RECV_TIMEOUT_SEC, 0 };
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
        handle_client(s, c);
        CLOSESOCK(c);
    }
    CLOSESOCK(srv);
    return 0;
}
