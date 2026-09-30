#define _CRT_SECURE_NO_WARNINGS
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
typedef SOCKET socket_t;
#define CLOSESOCK closesocket
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
typedef int socket_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define CLOSESOCK close
#endif

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PORT 8080
#define BUFFER_SIZE 16384

static void json_escape(const char *src, char *dst, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 7 < cap; ++i) {
        unsigned char c = (unsigned char)src[i];
        if (c == '"' || c == '\\') { dst[j++] = '\\'; dst[j++] = (char)c; }
        else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
        else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
        else if (c >= 32) dst[j++] = (char)c;
    }
    dst[j] = '\0';
}

static void lowercase_copy(const char *src, char *dst, size_t cap) {
    size_t i = 0;
    for (; src[i] && i + 1 < cap; ++i) dst[i] = (char)tolower((unsigned char)src[i]);
    dst[i] = '\0';
}

static void reply_for(const char *question, char *answer, size_t cap) {
    char q[2048]; lowercase_copy(question, q, sizeof q);
    if (!*q) snprintf(answer, cap, "Type a message and I’ll do my best to respond.");
    else if (strstr(q, "hello") || strstr(q, "hi") || strstr(q, "hey"))
        snprintf(answer, cap, "Hello! I’m Aster, an early C AI prototype. What would you like to explore?");
    else if (strstr(q, "who are you") || strstr(q, "what are you"))
        snprintf(answer, cap, "I’m Aster, a small local prototype written in C. Right now I use simple response rules, not a trained language model.");
    else if (strstr(q, "help") || strstr(q, "what can you do"))
        snprintf(answer, cap, "I can respond to a few basic prompts in this first prototype. The next milestone is adding a tokenizer and a small trained model.");
    else
        snprintf(answer, cap, "I received: “%.700s”\n\nI’m still a rule-based prototype, so I can’t reason about that yet. This chat UI is connected to the C server; the next step is giving the engine a real model.", question);
}

static void send_all(socket_t client, const char *data, size_t len) {
    while (len) {
        int n = send(client, data, (int)len, 0);
        if (n <= 0) return;
        data += n; len -= (size_t)n;
    }
}

static void send_response(socket_t client, const char *status, const char *type, const char *body) {
    char header[512]; size_t body_len = strlen(body);
    int n = snprintf(header, sizeof header,
        "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        status, type, body_len);
    send_all(client, header, (size_t)n); send_all(client, body, body_len);
}

static int serve_index(socket_t client) {
    FILE *f = fopen("index.html", "rb");
    if (!f) { send_response(client, "500 Internal Server Error", "text/plain; charset=utf-8", "Could not open index.html. Run the server from the project folder."); return 0; }
    fseek(f, 0, SEEK_END); long size = ftell(f); rewind(f);
    char *body = (char *)malloc((size_t)size + 1);
    if (!body) { fclose(f); return 0; }
    size_t got = fread(body, 1, (size_t)size, f); fclose(f); body[got] = '\0';
    send_response(client, "200 OK", "text/html; charset=utf-8", body); free(body); return 1;
}

static void handle_client(socket_t client) {
    char req[BUFFER_SIZE]; int n = recv(client, req, sizeof req - 1, 0);
    if (n <= 0) return; req[n] = '\0';
    if (strncmp(req, "GET / ", 6) == 0 || strncmp(req, "GET /index.html ", 16) == 0) { serve_index(client); return; }
    if (strncmp(req, "GET /api/status ", 16) == 0) {
        send_response(client, "200 OK", "application/json; charset=utf-8", "{\"status\":\"prototype\",\"engine\":\"C rule-based demo\"}"); return;
    }
    if (strncmp(req, "POST /api/chat ", 15) == 0) {
        char *body = strstr(req, "\r\n\r\n");
        if (!body) { send_response(client, "400 Bad Request", "application/json", "{\"error\":\"Invalid request\"}"); return; }
        body += 4;
        char question[2048] = {0};
        char *p = strstr(body, "\"message\"");
        if (p && (p = strchr(p, ':'))) {
            ++p; while (*p && isspace((unsigned char)*p)) ++p;
            if (*p == '"') {
                ++p; size_t j = 0;
                while (*p && *p != '"' && j + 1 < sizeof question) {
                    if (*p == '\\' && p[1]) { ++p; question[j++] = *p == 'n' ? '\n' : *p; }
                    else question[j++] = *p;
                    ++p;
                }
                question[j] = '\0';
            }
        }
        char answer[2048], escaped[4096], json[4608];
        reply_for(question, answer, sizeof answer); json_escape(answer, escaped, sizeof escaped);
        snprintf(json, sizeof json, "{\"reply\":\"%s\",\"engine\":\"C rule-based demo\"}", escaped);
        send_response(client, "200 OK", "application/json; charset=utf-8", json); return;
    }
    send_response(client, "404 Not Found", "text/plain; charset=utf-8", "Not found");
}

int main(void) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) { fprintf(stderr, "Winsock startup failed\n"); return 1; }
#endif
    socket_t server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == INVALID_SOCKET) { fprintf(stderr, "Could not create socket\n"); return 1; }
    struct sockaddr_in addr; memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET; addr.sin_port = htons(PORT); addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(server, (struct sockaddr *)&addr, sizeof addr) == SOCKET_ERROR || listen(server, 8) == SOCKET_ERROR) {
        fprintf(stderr, "Could not listen on 127.0.0.1:%d (is another copy running?)\n", PORT); CLOSESOCK(server); return 1;
    }
    printf("Aster is running at http://127.0.0.1:%d\nPress Ctrl+C to stop.\n", PORT); fflush(stdout);
    for (;;) {
        socket_t client = accept(server, NULL, NULL);
        if (client == INVALID_SOCKET) continue;
        handle_client(client); CLOSESOCK(client);
    }
}
