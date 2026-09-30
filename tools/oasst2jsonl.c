/* oasst2jsonl.c - convert an OpenAssistant (oasst1) export into Aster's chat
 * JSONL. A DATA PREP tool; deliberately not part of aster.exe.
 *
 * The project does no network access at any point and neither does this. You
 * download the dataset yourself, with whatever tooling you trust, and this
 * only reads a file that is already on disk. Nothing is fetched here.
 *
 *   oasst2jsonl --in oasst.jsonl --out aster_chat.jsonl [--lang en]
 *   oasst2jsonl --in oasst.jsonl --probe
 *
 * WHY THIS IS NOT A TRIVIAL FILTER
 *
 * oasst1 is a TREE, not a list of conversations. Every message carries a
 * message_id and a parent_message_id, and one user message can have several
 * assistant replies hanging off it. Aster's training format is strictly
 * linear: a system turn, then user/assistant pairs, nothing else. So this tool
 * reads the whole file, walks down from each root, and at every node that has
 * more than one child it keeps the FIRST and reports the rest as dropped.
 *
 * That is lossy, deliberately and visibly. Every count printed at the end says
 * how lossy. A converter that quietly dropped half the dataset and said
 * nothing would be worse than no converter at all.
 *
 * SCHEMA UNCERTAINTY IS A HARD FAILURE, NOT A SILENT ONE
 *
 * oasst1 has been published in more than one shape, and this program does not
 * know which one it has been handed. Rather than guess quietly, it recognises
 * records by their fields, reports what it managed to parse, and EXITS NON-ZERO
 * if it recognised nothing at all. If the file is not the format it expects,
 * the run fails and the --probe output shows the keys that are actually there.
 * An empty output file is never produced by accident.
 *
 * Build (from the project root):
 *   gcc -std=c11 -O2 -Wall -Wextra -I src -o oasst2jsonl.exe \
 *       tools/oasst2jsonl.c src/util.c src/jsonstr.c src/tokenizer.c
 */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jsonstr.h"
#include "tokenizer.h"
#include "util.h"

/* ------------------------------------------------------------------ */
/* Small JSON scanner.                                                 */
/*                                                                     */
/* oasst1 records are single-line objects, so a full parser would be  */
/* dead weight. What is actually needed is: find one key at one brace  */
/* depth and hand back its string value. Everything below is that.     */
/* ------------------------------------------------------------------ */

typedef struct {
    char  *p;
    size_t n;
    size_t cap;
} Buf;

static void buf_add(Buf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t c = b->cap ? b->cap : 65536;
        while (c < b->n + n + 1) c *= 2;
        b->p = (char *)aster_xrealloc(b->p, c);
        b->cap = c;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = '\0';
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t utf8_put(unsigned cp, char *out) {
    if (cp < 0x80u) { out[0] = (char)cp; return 1; }
    if (cp < 0x800u) {
        out[0] = (char)(0xC0u | (cp >> 6));
        out[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        out[0] = (char)(0xE0u | (cp >> 12));
        out[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (char)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    out[0] = (char)(0xF0u | (cp >> 18));
    out[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = (char)(0x80u | (cp & 0x3Fu));
    return 4;
}

/* If s[0] is '"', return the index just past the closing quote, else 0. */
static size_t scan_quoted(const char *s, size_t avail) {
    if (avail == 0 || s[0] != '"') return 0;
    for (size_t i = 1; i < avail; ++i) {
        if (s[i] == '\\') {
            if (i + 1 >= avail) return 0;
            ++i;
            continue;
        }
        if (s[i] == '"') return i + 1;
    }
    return 0;
}

/* Decode the JSON string starting at s[0] == '"', ending at index `end`
 * (just past the closing quote). Writes a NUL-terminated UTF-8 string.
 *   0 ok, -1 malformed, -2 out of room.
 *
 * `end` is an offset from s, NOT an absolute index into the caller's buffer,
 * because scan_quoted returns a relative one. Passing `s + i` together with an
 * absolute `i + end` reads i bytes too far, which silently runs each value into
 * the key after it. */
static int decode_string(const char *s, size_t end, char *out, size_t cap, size_t *outn) {
    size_t j = 0;
    for (size_t i = 1; i + 1 < end; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\\') {
            if (i + 2 >= end) return -1;
            char e = s[++i];
            switch (e) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u': {
                    if (i + 4 >= end) return -1;
                    unsigned cp = 0;
                    for (int k = 1; k <= 4; ++k) {
                        int d = hexval(s[i + k]);
                        if (d < 0) return -1;
                        cp = cp * 16u + (unsigned)d;
                    }
                    i += 4;
                    /* A high surrogate followed by a low one is one code
                     * point; a surrogate on its own is not valid UTF-8 and is
                     * rejected rather than passed through. */
                    if (cp >= 0xD800u && cp <= 0xDBFFu && i + 6 <= end - 2 &&
                        s[i + 1] == '\\' && s[i + 2] == 'u') {
                        unsigned lo = 0;
                        int ok = 1;
                        for (int k = 3; k <= 6; ++k) {
                            int d = hexval(s[i + k]);
                            if (d < 0) { ok = 0; break; }
                            lo = lo * 16u + (unsigned)d;
                        }
                        if (ok && lo >= 0xDC00u && lo <= 0xDFFFu) {
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                            i += 6;
                        }
                    }
                    if (cp >= 0xD800u && cp <= 0xDFFFu) return -1;
                    if (j + 4 > cap) return -2;
                    j += utf8_put(cp, out + j);
                    continue;
                }
                default: return -1;
            }
        }
        if (j + 1 > cap) return -2;
        out[j++] = (char)c;
    }
    out[j] = '\0';
    *outn = j;
    return 0;
}

/* Find `key` at brace depth `want` in one JSON object and copy its string
 * value into `out`.
 *   1 found (string copied), 0 absent, -1 present but not a string,
 *  -2 out of room, -3 the line is malformed. */
static int field_get(const char *s, size_t len, const char *key, int want,
                     char *out, size_t cap, size_t *outn) {
    int depth = 0;
    size_t i = 0;
    while (i < len) {
        char c = s[i];
        if (c == '"') {
            size_t end = scan_quoted(s + i, len - i);
            if (end == 0) return -3;
            size_t j = i + end;
            while (j < len && (s[j] == ' ' || s[j] == '\t')) ++j;
            if (j < len && s[j] == ':') {
                /* This string was a key. */
                char kbuf[128];
                size_t kn = 0;
                if (decode_string(s + i, end, kbuf, sizeof kbuf, &kn) != 0)
                    kbuf[0] = '\0';
                if (depth == want && kn != 0 && strcmp(kbuf, key) == 0) {
                    ++j;
                    while (j < len && (s[j] == ' ' || s[j] == '\t')) ++j;
                    if (j >= len) return -3;
                    if (s[j] != '"') { *outn = 0; return -1; }
                    size_t vend = scan_quoted(s + j, len - j);
                    if (vend == 0) return -3;
                    int rc = decode_string(s + j, vend, out, cap, outn);
                    /* decode_string answers 0 for success; this function
                     * promises 1. Returning rc directly would report every
                     * perfectly good field as "absent". */
                    return rc == 0 ? 1 : rc;
                }
                i = j + 1;
                continue;
            }
            i += end;
            continue;
        }
        if (c == '{' || c == '[') { ++depth; ++i; continue; }
        if (c == '}' || c == ']') { --depth; ++i; continue; }
        ++i;
    }
    return 0;
}

/* Print the keys at brace depth 1 of a record, so a schema mismatch can be
 * diagnosed instead of guessed at. */
static void list_keys(const char *s, size_t len) {
    int depth = 0;
    size_t i = 0;
    while (i < len) {
        char c = s[i];
        if (c == '"') {
            size_t end = scan_quoted(s + i, len - i);
            if (end == 0) return;
            size_t j = i + end;
            while (j < len && (s[j] == ' ' || s[j] == '\t')) ++j;
            if (j < len && s[j] == ':') {
                if (depth == 1) {
                    char kbuf[128];
                    size_t kn = 0;
                    if (decode_string(s + i, end, kbuf, sizeof kbuf, &kn) == 0)
                        printf("      %s\n", kbuf);
                }
                i = j + 1;
                continue;
            }
            i += end;
            continue;
        }
        if (c == '{' || c == '[') ++depth;
        else if (c == '}' || c == ']') --depth;
        ++i;
    }
}

/* ------------------------------------------------------------------ */
/* The message tree.                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    char  *id;      /* NULL if the record had no id field at all */
    char  *parent;  /* NULL for a root */
    char  *text;
    int    is_user; /* 1 = prompter, 0 = assistant */
} Msg;

typedef struct {
    Msg   *v;
    size_t n;
    size_t cap;
} MsgList;

static void msgs_push(MsgList *m, Msg x) {
    if (m->n == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 1024;
        m->v = (Msg *)aster_xrealloc(m->v, m->cap * sizeof(Msg));
    }
    m->v[m->n++] = x;
}

/* Children are looked up by binary search, so the messages are sorted by
 * parent string and then by original position. The position tiebreak is what
 * makes "keep the first child" deterministic: qsort alone would not preserve
 * input order within a group.
 *
 * The comparator reads a file-scope pointer rather than taking a context
 * argument because qsort_r has two incompatible signatures (glibc and
 * MinGW-w64 disagree on the argument order). Single-threaded, so this is safe
 * and portable. */
static const Msg *g_sort_msgs;

static int cmp_by_parent(const void *a, const void *b) {
    size_t ia = *(const size_t *)a, ib = *(const size_t *)b;
    const char *pa = g_sort_msgs[ia].parent ? g_sort_msgs[ia].parent : "";
    const char *pb = g_sort_msgs[ib].parent ? g_sort_msgs[ib].parent : "";
    int c = strcmp(pa, pb);
    if (c != 0) return c;
    return ia < ib ? -1 : (ia > ib ? 1 : 0);
}

/* Half-open range of `order` holding children of `parent`. */
static void child_range(const size_t *order, size_t n, const char *parent,
                        size_t *lo, size_t *hi) {
    const char *key = parent ? parent : "";
    size_t a = 0, b = n;
    while (a < b) {
        size_t mid = a + (b - a) / 2;
        const char *p = g_sort_msgs[order[mid]].parent
                            ? g_sort_msgs[order[mid]].parent : "";
        if (strcmp(p, key) < 0) a = mid + 1; else b = mid;
    }
    *lo = a;
    b = n;
    while (a < b) {
        size_t mid = a + (b - a) / 2;
        const char *p = g_sort_msgs[order[mid]].parent
                            ? g_sort_msgs[order[mid]].parent : "";
        if (strcmp(p, key) <= 0) a = mid + 1; else b = mid;
    }
    *hi = a;
}

/* ------------------------------------------------------------------ */

static void usage(void) {
    printf(
        "oasst2jsonl - OpenAssistant (oasst1) export -> Aster chat JSONL\n"
        "\n"
        "  oasst2jsonl --in FILE --out FILE [options]\n"
        "  oasst2jsonl --in FILE --probe\n"
        "\n"
        "Options:\n"
        "  --in FILE        the oasst1 export to read (JSON Lines)\n"
        "  --out FILE       where to write Aster-format JSONL\n"
        "  --lang CODE      keep only this language (e.g. en). Default: keep all\n"
        "  --min-chars N    drop turns shorter than N bytes. Default: 8\n"
        "  --max-turns N    keep at most N turns per conversation. Default: 6\n"
        "  --probe          print the record layout and exit; writes nothing\n"
        "\n"
        "This tool reads a local file. It does not download anything.\n");
}

/* Try each candidate key in turn and return the first one that is present. */
static int get_any(const char *line, size_t len, const char *const *keys,
                   int nkeys, int depth, char *out, size_t cap, size_t *outn) {
    for (int i = 0; i < nkeys; ++i) {
        int rc = field_get(line, len, keys[i], depth, out, cap, outn);
        if (rc == 1 || rc == -1) return rc;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *in = NULL, *out = NULL, *lang = NULL;
    int probe = 0;
    size_t min_chars = 8, max_turns = 6;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc)            in = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)      out = argv[++i];
        else if (!strcmp(argv[i], "--lang") && i + 1 < argc)     lang = argv[++i];
        else if (!strcmp(argv[i], "--min-chars") && i + 1 < argc) min_chars = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-turns") && i + 1 < argc) max_turns = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe"))                    probe = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n\n", argv[i]); usage(); return 2; }
    }
    if (!in) { fprintf(stderr, "--in is required\n\n"); usage(); return 2; }
    if (!probe && !out) { fprintf(stderr, "--out is required (or use --probe)\n\n"); usage(); return 2; }
    if (max_turns == 0) max_turns = 1;

    int ok = 0;
    size_t flen = 0;
    char *file = aster_read_file(in, &flen, &ok);
    if (!ok) aster_fail("could not read %s", in);
    if (flen == 0) aster_fail("%s is empty", in);

    /* -- probe: show what the first few records actually look like. */
    if (probe) {
        size_t li = 0, shown = 0;
        while (li < flen && shown < 3) {
            size_t lj = li;
            while (lj < flen && file[lj] != '\n') ++lj;
            size_t n = lj - li;
            while (n > 0 && (file[li + n - 1] == '\r')) --n;
            if (n > 1) {
                printf("  record %zu (%zu bytes) top-level keys:\n", shown + 1, n);
                list_keys(file + li, n);
                shown++;
            }
            li = lj + 1;
        }
        printf("\n  If \"text\" and \"role\" are not in that list, this is not the\n"
               "  oasst1 layout this tool understands, and a real run would stop\n"
               "  with an error rather than write a wrong file.\n");
        free(file);
        return 0;
    }

    static const char *const ID_KEYS[]   = { "message_id", "_id", "id" };
    static const char *const TEXT_KEYS[] = { "text", "value" };

    MsgList msgs = { NULL, 0, 0 };
    size_t lines = 0, parsed = 0, no_id = 0, wrong_lang = 0;
    size_t bad_role = 0, bad_text = 0, bad_utf8 = 0, too_short = 0;
    size_t li = 0;

    while (li < flen) {
        size_t lj = li;
        while (lj < flen && file[lj] != '\n') ++lj;
        size_t n = lj - li;
        while (n > 0 && file[li + n - 1] == '\r') --n;
        if (n > 1) {
            lines++;
            const char *line = file + li;
            char v[65536];
            size_t vn = 0;

            if (lang) {
                int rc = field_get(line, n, "lang", 1, v, sizeof v, &vn);
                if (rc == 1 && strcmp(v, lang) != 0) wrong_lang++;
                else if (rc == 1) { /* keep */ }
                else if (rc == 0) wrong_lang++;   /* no language: cannot confirm */
                if (rc == 1 && strcmp(v, lang) != 0) { li = lj + 1; continue; }
                if (rc == 0) { li = lj + 1; continue; }
            }

            int rr = field_get(line, n, "role", 1, v, sizeof v, &vn);
            if (rr != 1) { bad_role++; li = lj + 1; continue; }
            int is_user = (!strcmp(v, "prompter") || !strcmp(v, "user"));
            if (!is_user && strcmp(v, "assistant") != 0) { bad_role++; li = lj + 1; continue; }

            /* The text is either a top-level "text" or one nested inside a
             * "content" object. Try both; the depth is what distinguishes
             * them. */
            int tr = get_any(line, n, TEXT_KEYS, 2, 1, v, sizeof v, &vn);
            if (tr != 1) tr = get_any(line, n, TEXT_KEYS, 2, 2, v, sizeof v, &vn);
            if (tr != 1) { bad_text++; li = lj + 1; continue; }

            char *text = (char *)aster_xmalloc(vn + 1);
            memcpy(text, v, vn + 1);

            /* Aster's loader rejects invalid UTF-8 rather than repairing it,
             * so rejecting here gives a clear reason instead of a later
             * "invalid JSONL" error pointing at the wrong file. */
            if (!utf8_is_valid(text, vn)) { bad_utf8++; free(text); li = lj + 1; continue; }
            if (vn < min_chars) { too_short++; free(text); li = lj + 1; continue; }

            char idbuf[256];
            size_t idn = 0;
            Msg m;
            memset(&m, 0, sizeof m);
            m.text = text;
            m.is_user = is_user;

            if (get_any(line, n, ID_KEYS, 3, 1, idbuf, sizeof idbuf, &idn) == 1) {
                m.id = (char *)aster_xmalloc(idn + 1);
                memcpy(m.id, idbuf, idn + 1);
            } else {
                no_id++;
                m.id = NULL;
            }

            char pbuf[256];
            size_t pn = 0;
            int pr = field_get(line, n, "parent_message_id", 1, pbuf, sizeof pbuf, &pn);
            if (pr == 1) {
                m.parent = (char *)aster_xmalloc(pn + 1);
                memcpy(m.parent, pbuf, pn + 1);
            } else {
                /* absent, null, or a non-string: this is a root. */
                m.parent = NULL;
            }
            msgs_push(&msgs, m);
            parsed++;
        }
        li = lj + 1;
    }
    free(file);

    aster_info("read %s", in);
    aster_info("  lines %zu, parsed as messages %zu", lines, parsed);
    if (wrong_lang)  aster_info("  skipped: language filter (%zu)", wrong_lang);
    if (no_id)       aster_warn("%zu records had no id field; their chains cannot be "
                                "followed and will not appear in the output", no_id);
    if (bad_role)    aster_info("  skipped: unrecognised role (%zu)", bad_role);
    if (bad_text)    aster_info("  skipped: no readable text (%zu)", bad_text);
    if (bad_utf8)    aster_info("  skipped: invalid UTF-8 (%zu)", bad_utf8);
    if (too_short)   aster_info("  skipped: shorter than --min-chars (%zu)", too_short);

    if (parsed == 0) {
        aster_fail(
            "no records were recognised. This file is not the oasst1 layout this\n"
            "  tool understands, so nothing was written. Run with --probe to see\n"
            "  the keys that are actually present. Writing an empty training file\n"
            "  would be worse than stopping here.");
    }

    /* Index messages by parent. */
    size_t n = msgs.n;
    size_t *order = (size_t *)aster_xmalloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) order[i] = i;
    g_sort_msgs = msgs.v;
    qsort(order, n, sizeof(size_t), cmp_by_parent);

    Buf buf = { NULL, 0, 0 };
    buf_add(&buf, "", 0);

    size_t roots = 0, written = 0, turns_written = 0;
    size_t branches_dropped = 0, chain_too_long = 0;
    size_t bad_alternation = 0, unterminated = 0, no_root = 0;

    size_t *chain = (size_t *)aster_xmalloc(max_turns * sizeof(size_t));

    for (size_t i = 0; i < n; ++i) {
        if (msgs.v[i].parent != NULL) continue;   /* not a root */
        if (!msgs.v[i].is_user) continue;          /* a root must open a thread */
        roots++;

        size_t len = 0, cur = i;
        int truncated = 0;

        /* The root IS the first turn of the conversation. Pushing only the
         * children would start every emitted chain at its first reply, and
         * Aster's loader rejects an assistant turn that does not follow a user
         * one. */
        chain[len++] = i;

        for (;;) {
            size_t lo, hi;
            child_range(order, n, msgs.v[cur].id, &lo, &hi);
            if (lo == hi) break;                   /* leaf */

            size_t nchild = hi - lo;
            if (nchild > 1) branches_dropped += nchild - 1;

            if (len == max_turns) { truncated = 1; break; }
            chain[len++] = order[lo];
            cur = order[lo];
        }
        if (truncated) chain_too_long++;
        if (len < 2) { no_root++; continue; }   /* a root with no reply */

        /* Aster requires a strictly alternating user/assistant chain. */
        int ok_chain = 1;
        for (size_t k = 1; k < len; ++k)
            if (msgs.v[chain[k]].is_user == msgs.v[chain[k - 1]].is_user)
                { ok_chain = 0; break; }
        if (!ok_chain) { bad_alternation++; continue; }
        if (msgs.v[chain[len - 1]].is_user) { unterminated++; continue; }

        /* Emit. A new system turn starts a new training conversation, so the
         * multi-turn chain below is one conversation with several pairs. */
        buf_add(&buf, "{\"role\":\"system\",\"content\":\"\"}\n", 31);
        for (size_t k = 0; k < len; ++k) {
            const char *role = msgs.v[chain[k]].is_user ? "user" : "assistant";
            char *head = (char *)aster_xmalloc(64);
            int hn = snprintf(head, 64, "{\"role\":\"%s\",\"content\":\"", role);
            buf_add(&buf, head, (size_t)hn);
            free(head);

            const char *text = msgs.v[chain[k]].text;
            size_t tn = strlen(text);
            char *esc = (char *)aster_xmalloc(tn * 6 + 16);
            if (json_escape(text, esc, tn * 6 + 16) < 0) {
                free(esc);
                bad_text++;
                continue;
            }
            buf_add(&buf, esc, strlen(esc));
            buf_add(&buf, "\"}\n", 3);
            free(esc);
            turns_written++;
        }
        written++;
    }

    free(chain);
    free(order);
    for (size_t i = 0; i < msgs.n; ++i) {
        free(msgs.v[i].id);
        free(msgs.v[i].parent);
        free(msgs.v[i].text);
    }
    free(msgs.v);

    aster_info("  roots found %zu, conversations written %zu", roots, written);
    aster_info("  turns written %zu, total %zu bytes", turns_written, buf.n);
    if (branches_dropped)
        aster_warn("%zu alternative replies were dropped: oasst1 is a tree and Aster's "
                   "format is linear, so one reply per prompt is kept", branches_dropped);
    if (chain_too_long)  aster_info("  truncated at --max-turns: %zu chains", chain_too_long);
    if (bad_alternation) aster_info("  dropped, roles did not alternate: %zu", bad_alternation);
    if (unterminated)    aster_info("  dropped, last turn had no reply: %zu", unterminated);
    if (no_root)         aster_info("  dropped, root had no child: %zu", no_root);

    if (written == 0) {
        free(buf.p);
        aster_fail("no conversation survived the filters, so nothing was written");
    }

    aster_mkdir_for_file(out);
    aster_write_file_atomic(out, buf.p, buf.n);
    free(buf.p);
    aster_info("wrote %s", out);

    aster_info("");
    aster_info("Before training on this:");
    aster_info("  1. READ IT. It is other people's writing. Look for names, contact");
    aster_info("     details, and anything personal, and remove what you should not");
    aster_info("     keep. Do not assume the dataset was scrubbed for you.");
    aster_info("  2. Record the licence and the SHA-256 in data/README.md, and add");
    aster_info("     the file to data/manifest.json.");
    aster_info("  3. Keep the validation set DISJOINT from this one.");
    return 0;
}
