#include "jsonstr.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

static void utf8_put(char **dst, unsigned cp) {
    char *d = *dst;
    if (cp < 0x80) {
        *d++ = (char)cp;
    } else if (cp < 0x800) {
        *d++ = (char)(0xC0 | (cp >> 6));
        *d++ = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *d++ = (char)(0xE0 | (cp >> 12));
        *d++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *d++ = (char)(0x80 | (cp & 0x3F));
    } else {
        *d++ = (char)(0xF0 | (cp >> 18));
        *d++ = (char)(0x80 | ((cp >> 12) & 0x3F));
        *d++ = (char)(0x80 | ((cp >> 6) & 0x3F));
        *d++ = (char)(0x80 | (cp & 0x3F));
    }
    *dst = d;
}

static int hex4(const char *s, unsigned *out) {
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return -1;
    }
    *out = v;
    return 0;
}

/* Decode a JSON string starting at body[i] == '"'. Writes the unescaped value
 * to out (which may be NULL when the caller only wants to skip it) and
 * advances *i past the closing quote. */
static int decode_string(const char *body, size_t len, size_t *i,
                         char *out, size_t cap, size_t *outn,
                         char *err, size_t errlen) {
    size_t k = *i + 1;
    size_t j = 0;
    int overflow = 0;

    while (k < len && body[k] != '"') {
        unsigned char c = (unsigned char)body[k];
        char ch;
        if (c == '\\') {
            if (k + 1 >= len) { snprintf(err, errlen, "string ends in a backslash"); return -1; }
            char e = body[++k];
            switch (e) {
                case '"':  ch = '"';  break;
                case '\\': ch = '\\'; break;
                case '/':  ch = '/';  break;
                case 'b':  ch = '\b'; break;
                case 'f':  ch = '\f'; break;
                case 'n':  ch = '\n'; break;
                case 'r':  ch = '\r'; break;
                case 't':  ch = '\t'; break;
                case 'u': {
                    unsigned cp;
                    if (k + 4 >= len || hex4(body + k + 1, &cp) != 0) {
                        snprintf(err, errlen, "malformed \\u escape"); return -1;
                    }
                    k += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {          /* high surrogate */
                        unsigned lo;
                        if (k + 6 >= len || body[k + 1] != '\\' || body[k + 2] != 'u' ||
                            hex4(body + k + 3, &lo) != 0 || lo < 0xDC00 || lo > 0xDFFF) {
                            snprintf(err, errlen, "unpaired UTF-16 surrogate"); return -1;
                        }
                        k += 6;
                        cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        snprintf(err, errlen, "unpaired UTF-16 surrogate"); return -1;
                    }
                    if (out) {
                        if (j + 5 < cap) { char *p = out + j; utf8_put(&p, cp); j = (size_t)(p - out); }
                        else overflow = 1;
                    }
                    ++k;
                    continue;
                }
                default: snprintf(err, errlen, "unknown escape \\%c", e); return -1;
            }
            ++k;
        } else if (c < 0x20) {
            snprintf(err, errlen, "raw control character 0x%02X inside a string", c);
            return -1;
        } else {
            ch = (char)c;
            ++k;
        }
        if (out) {
            if (j + 1 < cap) out[j] = ch; else overflow = 1;
        }
        ++j;
    }
    if (k >= len) { snprintf(err, errlen, "unterminated string"); return -1; }
    if (out) {
        if (overflow) return -3;
        out[j < cap ? j : cap - 1] = '\0';
    }
    if (outn) *outn = j;
    *i = k + 1;
    return 0;
}

int json_get_string(const char *body, size_t len, const char *key,
                    char *out, size_t cap, char *err, size_t errlen) {
    size_t i = 0;
    char keybuf[128];
    int found = 0;

    while (i < len && is_ws(body[i])) ++i;
    if (i >= len || body[i] != '{') { snprintf(err, errlen, "body is not a JSON object"); return -1; }
    ++i;

    while (1) {
        while (i < len && is_ws(body[i])) ++i;
        if (i < len && body[i] == '}') { ++i; break; }
        if (i >= len || body[i] != '"') { snprintf(err, errlen, "expected a quoted key at byte %zu", i); return -1; }

        int rc = decode_string(body, len, &i, keybuf, sizeof keybuf, NULL, err, errlen);
        if (rc == -3) { snprintf(err, errlen, "key is too long"); return -1; }
        if (rc != 0) return -1;

        while (i < len && is_ws(body[i])) ++i;
        if (i >= len || body[i] != ':') { snprintf(err, errlen, "expected ':' after a key"); return -1; }
        ++i;
        while (i < len && is_ws(body[i])) ++i;
        if (i >= len) { snprintf(err, errlen, "object ends after ':'"); return -1; }

        int match = (strcmp(keybuf, key) == 0);
        if (i < len && body[i] == '"') {
            if (match) {
                if (found) { snprintf(err, errlen, "duplicate key '%s'", key); return -1; }
                rc = decode_string(body, len, &i, out, cap, NULL, err, errlen);
                if (rc == -3) return -3;
                if (rc != 0) return -1;
                found = 1;
            } else {
                rc = decode_string(body, len, &i, NULL, 0, NULL, err, errlen);
                if (rc != 0) return -1;
            }
        } else {
            /* Non-string values are skipped so a client sending
             * {"temperature":0.7,"message":"hi"} still works. */
            size_t start = i;
            while (i < len && body[i] != ',' && body[i] != '}' && !is_ws(body[i])) ++i;
            if (i == start) { snprintf(err, errlen, "empty value at byte %zu", start); return -1; }
            if (match) { snprintf(err, errlen, "value of '%s' is not a string", key); return -1; }
        }

        while (i < len && is_ws(body[i])) ++i;
        if (i < len && body[i] == ',') { ++i; continue; }
        if (i < len && body[i] == '}') { ++i; break; }
        snprintf(err, errlen, "expected ',' or '}' at byte %zu", i);
        return -1;
    }

    while (i < len && is_ws(body[i])) ++i;
    if (i != len) { snprintf(err, errlen, "unexpected trailing content"); return -1; }
    return found ? 0 : -2;
}

int json_escape(const char *in, char *out, size_t cap) {
    size_t j = 0;
    static const char *hex = "0123456789abcdef";
    for (size_t i = 0; in[i]; ++i) {
        unsigned char c = (unsigned char)in[i];
        const char *rep = NULL;
        char buf[8];
        switch (c) {
            case '"':  rep = "\\\""; break;
            case '\\': rep = "\\\\"; break;
            case '\b': rep = "\\b";  break;
            case '\f': rep = "\\f";  break;
            case '\n': rep = "\\n";  break;
            case '\r': rep = "\\r";  break;
            case '\t': rep = "\\t";  break;
            default:
                if (c < 0x20) {
                    buf[0] = '\\'; buf[1] = 'u'; buf[2] = '0'; buf[3] = '0';
                    buf[4] = hex[(c >> 4) & 0xF]; buf[5] = hex[c & 0xF];
                    buf[6] = '\0';
                    rep = buf;
                }
                break;
        }
        if (rep) {
            size_t n = strlen(rep);
            if (j + n + 1 > cap) return -1;
            memcpy(out + j, rep, n);
            j += n;
        } else {
            if (j + 2 > cap) return -1;
            out[j++] = (char)c;
        }
    }
    if (cap == 0) return -1;
    out[j < cap ? j : cap - 1] = '\0';
    return (j < cap) ? 0 : -1;
}
