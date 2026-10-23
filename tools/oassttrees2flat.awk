# oassttrees2flat.awk - flatten the oasst1 *.trees.jsonl.gz export into the flat
# message-per-line form that tools/oasst2jsonl.c reads.
#
# WHY THIS EXISTS
# ---------------
# oasst1 ships the same conversation data twice, and the two files are not
# interchangeable:
#
#   *.messages.jsonl  88 838 records, one per message, each carrying
#                      message_tree_id and created_date -- but NO
#                      parent_message_id. Zero of 88 838 records have one.
#   *.trees.jsonl     10 364 lines, one per conversation tree, each holding
#                      the SAME messages nested under "children", with the
#                      parent/child relationship implied by the nesting.
#
# The tree structure is the whole point of the dataset: one user message can
# have several assistant replies, and that branching is exactly what a linear
# training format has to flatten away. The messages file discards the structure
# without recording it, so parentage cannot be recovered from it. The trees
# file keeps it, and this script reads it.
#
# It is a data-prep step, not part of aster.exe, and it downloads nothing.
#
# THE PART THAT IS EASY TO GET WRONG
# ----------------------------------
# A message object is not a leaf. It carries "labels", "rank", "review_state"
# and other nested objects, so the message's own closing brace is not the first
# '}' seen after its "message_id". Any logic that treats the first '}' as the end
# of the message closes the record early, and -- because the next message's
# "message_id" then arrives while the previous record still looks open -- never
# captures the rest of the tree. One record out, silently.
#
# The fix is to stop reasoning about which brace ends which message and index
# the open messages by brace depth instead. A '{' raises the depth, a '}' lowers
# it, and the object that just closed is always the one that was at the NEW
# depth plus one. A message read at depth D therefore closes on the '}' that
# brings the depth to D-1, and every nested '}' inside it is at a depth above D
# and cannot touch it.
#
# That also makes the parent lookup trivial and exact: the parent of a message
# at depth D is simply the nearest shallower depth that still holds a message
# id. Ancestors are still open when the child is read, so nothing has to be
# popped -- the depth index is the stack.
#
# The escape flag is not optional either: oasst1 text contains quotes,
# backslashes and newlines, and a naive scan desynchronises on the first one
# and silently corrupts every message after it.

function esc_json(s) {
    # The text came out of JSON, so it may hold raw newlines and control
    # characters. A literal newline inside a JSONL record splits one record
    # into two, and the second is a syntax error.
    gsub(/\\/, "\\\\", s)
    gsub(/"/, "\\\"", s)
    gsub(/\t/, "\\t", s)
    gsub(/\r/, "\\r", s)
    gsub(/\n/, "\\n", s)
    return s
}

BEGIN { FS = "" }

{
    line     = $0
    len      = length(line)
    depth    = 0
    trees++
    in_str   = 0
    esc      = 0
    key      = ""
    captured = 0

    for (i = 1; i <= len; i++) {
        c = substr(line, i, 1)

        if (in_str) {
            if (esc)              { buf = buf c; esc = 0 }
            else if (c == "\\")   { buf = buf c; esc = 1 }
            else if (c == "\"")   {
                in_str = 0
                # Key or value? Skip whitespace after the closing quote; a ':'
                # there means this string was a key.
                j = i + 1
                while (j <= len && substr(line, j, 1) == " ") j++
                if (j <= len && substr(line, j, 1) == ":") {
                    key = buf
                } else {
                    value(key, buf, depth)
                    key = ""
                }
                buf = ""
            }
            else buf = buf c
            continue
        }

        if (c == "\"") { in_str = 1; buf = ""; esc = 0; continue }

        if (c == "{") { depth++; clear(depth); continue }

        if (c == "}") {
            # The object that just closed is the one at the CURRENT depth; the
            # brace takes it back down to depth-1.
            if (id_at[depth] != "") {
                emit(depth)
                clear(depth)
            }
            depth--
            continue
        }
    }
    # A well-formed line leaves nothing open, but flush rather than drop it if
    # the last message somehow ran to end-of-line.
    while (depth >= 0) {
        if (id_at[depth] != "") { emit(depth); clear(depth) }
        depth--
    }
    next
}

# One JSON value finished. Record it against the innermost open message.
function value(k, v, d) {
    if (k == "message_id") {
        # Exact match, so the enclosing "message_tree_id" at the top of the
        # tree object cannot be mistaken for it.
        id_at[d]   = v
        role_at[d] = ""
        lang_at[d] = ""
        text_at[d] = ""
        captured++
        return
    }
    if (id_at[d] == "") return          # not inside a message object
    if      (k == "role") role_at[d] = v
    else if (k == "lang") lang_at[d] = v
    else if (k == "text") text_at[d] = v
}

function clear(d) {
    id_at[d]   = ""
    role_at[d] = ""
    lang_at[d] = ""
    text_at[d] = ""
}

# Write one flat record. The parent is the nearest shallower message still
# open, which is the conversation parent because the nesting is the tree.
function emit(d,   parent, k) {
    parent = ""
    for (k = d - 1; k >= 0; k--) {
        if (id_at[k] != "") { parent = id_at[k]; break }
    }
    # A root OMITS parent_message_id rather than setting it to "". The converter
    # reads an absent or null value as a root; an empty string is a string, so
    # it would be taken for a parent whose id is "" and no root would ever be
    # found. The published messages file omits the key for the same reason.
    if (parent == "")
        printf "{\"message_id\":\"%s\",\"role\":\"%s\",\"lang\":\"%s\",\"text\":\"%s\"}\n",
               id_at[d], esc_json(role_at[d]), esc_json(lang_at[d]), esc_json(text_at[d])
    else
        printf "{\"message_id\":\"%s\",\"parent_message_id\":\"%s\",\"role\":\"%s\",\"lang\":\"%s\",\"text\":\"%s\"}\n",
               id_at[d], parent, esc_json(role_at[d]), esc_json(lang_at[d]), esc_json(text_at[d])
    emitted++
}

END {
    printf "# trees %d  messages %d\n", trees, emitted > "/dev/stderr"
}
