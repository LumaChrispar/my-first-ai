#include "tokenizer.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

void token_list_init(TokenList *tl) {
    tl->ids = NULL; tl->count = 0; tl->cap = 0; tl->overflowed = 0;
}

void token_list_free(TokenList *tl) {
    free(tl->ids);
    token_list_init(tl);
}

void token_list_push(TokenList *tl, uint16_t id) {
    if (tl->count == tl->cap) {
        tl->cap = tl->cap ? tl->cap * 2 : 256;
        tl->ids = (uint16_t *)aster_xrealloc(tl->ids, aster_size_mul(tl->cap, sizeof(uint16_t)));
    }
    tl->ids[tl->count++] = id;
}

void token_list_push_many(TokenList *tl, const uint16_t *ids, size_t n) {
    for (size_t i = 0; i < n; ++i) token_list_push(tl, ids[i]);
}

int tok_encode(const char *text, size_t len, TokenList *out) {
    out->overflowed = 0;
    for (size_t i = 0; i < len; ++i) {
        /* NUL bytes are stripped during data cleaning, but guard anyway so a
         * stray byte can never become a terminator inside the model. */
        if (text[i] == '\0') continue;
        token_list_push(out, (uint16_t)(unsigned char)text[i]);
    }
    return out->overflowed ? -1 : 0;
}

size_t tok_decode(const uint16_t *ids, size_t n, char *out, size_t cap, int end_marker_newline) {
    if (cap == 0) return 0;
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
        char c;
        if (ids[i] <= ASTER_BYTE_MAX) {
            c = (char)(unsigned char)ids[i];
        } else if (ids[i] == TOK_END) {
            c = end_marker_newline ? '\n' : '\0';
        } else {
            continue;  /* SYSTEM / USER / ASSISTANT / PAD are structural */
        }
        if (j + 1 < cap) out[j++] = c;
    }
    out[j] = '\0';
    return j;
}

const char *tok_name(uint16_t id) {
    static const char *names[] = {"SYSTEM", "USER", "ASSISTANT", "END", "PAD"};
    if (id <= ASTER_BYTE_MAX) return "byte";
    if (id >= TOK_SYSTEM && id <= TOK_PAD) return names[id - TOK_SYSTEM];
    return "invalid";
}

/* ------------------------------------------------------------- utf-8 ---- */
/* Length of the UTF-8 sequence starting at in[0], or 0 if it is not a valid
 * lead byte. Rejects overlong forms, surrogates, and out-of-range code points
 * so that sanitize() is actually correct rather than merely tolerant. */
static int utf8_seq_len(const unsigned char *in) {
    unsigned char c = in[0];
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) return 2;
    if (c >= 0xE0 && c <= 0xEF) return 3;
    if (c >= 0xF0 && c <= 0xF4) return 4;
    return 0;
}

static int utf8_seq_valid(const unsigned char *in, int len) {
    for (int k = 1; k < len; ++k) if ((in[k] & 0xC0) != 0x80) return 0;
    if (len == 2) return 1;
    if (len == 3) {
        if (in[0] == 0xE0 && in[1] < 0xA0) return 0;          /* overlong */
        if (in[0] == 0xED && in[1] >= 0xA0) return 0;         /* surrogate */
        return 1;
    }
    if (in[0] == 0xF0 && in[1] < 0x90) return 0;              /* overlong */
    if (in[0] == 0xF4 && in[1] >= 0x90) return 0;             /* > U+10FFFF */
    return 1;
}

int utf8_sanitize(const char *in, size_t len, char *out, size_t cap) {
    if (cap == 0) return 0;
    const unsigned char *p = (const unsigned char *)in;
    size_t i = 0, j = 0;
    int replaced = 0;
    while (i < len) {
        int sl = utf8_seq_len(p + i);
        if (sl > 0 && (i + (size_t)sl <= len) && utf8_seq_valid(p + i, sl)) {
            if (j + (size_t)sl + 1 < cap) { memcpy(out + j, p + i, (size_t)sl); j += (size_t)sl; }
            i += (size_t)sl;
        } else {
            replaced = 1;
            if (j + 4 < cap) { memcpy(out + j, "\xEF\xBF\xBD", 3); j += 3; }
            i += 1;
        }
    }
    out[j] = '\0';
    return replaced;
}

int utf8_is_valid(const char *in, size_t len) {
    const unsigned char *p = (const unsigned char *)in;
    size_t i = 0;
    while (i < len) {
        int sl = utf8_seq_len(p + i);
        if (sl == 0 || i + (size_t)sl > len || !utf8_seq_valid(p + i, sl)) return 0;
        i += (size_t)sl;
    }
    return 1;
}
