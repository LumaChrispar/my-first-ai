/* jsonstr.h - the small amount of JSON handling this project actually needs.
 *
 * The chat API accepts one flat object and the instruction-tuning data is a
 * flat object per line, so a full parser would be dead weight. These helpers
 * do validate the object structure they walk, which is what lets the server
 * reject malformed bodies instead of guessing.
 */
#ifndef ASTER_JSONSTR_H
#define ASTER_JSONSTR_H

#include <stddef.h>

/* Copy the string value of `key` out of the JSON object in [body, body+len)
 * into `out`, unescaping as it goes.
 *   0  success
 *  -1  the object is malformed (details written to err)
 *  -2  the key is absent
 *  -3  the decoded value does not fit in cap */
int json_get_string(const char *body, size_t len, const char *key,
                    char *out, size_t cap, char *err, size_t errlen);

/* Escape a byte string as a JSON string body (no surrounding quotes).
 * Control characters become \u00XX so the result is always valid JSON.
 *   0  success
 *  -1  the escaped form did not fit in cap */
int json_escape(const char *in, char *out, size_t cap);

#endif /* ASTER_JSONSTR_H */
