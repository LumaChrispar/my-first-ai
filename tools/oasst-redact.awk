# oasst-redact.awk - remove personal data from a converted oasst1 chat JSONL.
#
# WHY
# ---
# oasst1 was written by volunteers, so the converted corpus contains other
# people's email addresses, phone numbers and similar identifiers. The build
# spec excludes personal records from training data, and a file that has been
# downloaded is not thereby scrubbed. The converter prints this warning after
# every run; this is the step that acts on it.
#
# WHAT IT DOES
# ------------
# It redacts rather than deletes. Dropping a whole conversation because it
# contains one address loses the other 2 000 characters of it, and a corpus
# that is missing exactly the conversations where someone described their own
# situation is a biased corpus. Replacing the identifier keeps the text.
#
# The replacements are deliberately plain ASCII with no quote or backslash, so
# they cannot break the JSON they sit inside.
#
# WHAT IT DOES NOT CATCH
# ---------------------
# Real names, employers, towns, health details, and anything else a person
# wrote about themselves. Those cannot be found by pattern and need a human to
# read the file. This narrows the problem; it does not close it.
#
# It is a data-prep step. It is not part of aster.exe and downloads nothing.

BEGIN {
    n_email = n_num = n_ip = 0
}

{
    # Email addresses.
    while (match($0, /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/)) {
        $0 = substr($0, 1, RSTART - 1) "[email-redacted]" substr($0, RSTART + RLENGTH)
        n_email++
    }
    # IPv4 addresses, before the number rule so they are not half-eaten.
    while (match($0, /([0-9]{1,3}\.){3}[0-9]{1,3}/)) {
        $0 = substr($0, 1, RSTART - 1) "[ip-redacted]" substr($0, RSTART + RLENGTH)
        n_ip++
    }
    # Runs of digits with optional separators: phone numbers, account numbers,
    # postcodes. Deliberately over-broad -- it will also catch a long date or
    # a measurement, and losing one of those is the cheap direction to err in.
    while (match($0, /[0-9][0-9 ().-]{8,}[0-9]/)) {
        $0 = substr($0, 1, RSTART - 1) "[number-redacted]" substr($0, RSTART + RLENGTH)
        n_num++
    }
    print
}

END {
    printf "# redacted: %d email, %d ip, %d number runs\n",
           n_email, n_ip, n_num > "/dev/stderr"
}
