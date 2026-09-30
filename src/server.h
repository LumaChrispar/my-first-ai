/* server.h - loopback HTTP server for the local chat interface. */
#ifndef ASTER_SERVER_H
#define ASTER_SERVER_H

#include <stdint.h>
#include "model.h"

typedef struct {
    AsterModel *model;          /* NULL means "not trained" and every chat
                                 * request is answered with 503 */
    char  name[192];            /* file name only; local paths are never shown */
    uint32_t step;
    uint32_t seed;
    int    max_new_tokens;
    float  temperature;
    int    top_k;
    /* Text between the SYSTEM and USER markers. Must match the system text
     * the model was trained with, or be empty if the corpus had none. */
    const char *system;
} AsterServer;

/* Binds and serves until interrupted. `host` must be a loopback address;
 * anything else is refused unless the caller has already done its own
 * explicit opt-in. Returns 0 on a clean shutdown, non-zero on failure. */
int aster_serve(AsterServer *s, const char *host, int port);

#endif /* ASTER_SERVER_H */
