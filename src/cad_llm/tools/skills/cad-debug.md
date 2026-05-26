# CAD debug (sandbox failures)

When sandbox returns non-zero:

1. **read_file** — the failing file and traceback line.
2. **search_cadquery_docs** — at most once for the user's goal, not the raw error string.
3. **write_file** or **search_replace** — fix using documented APIs only.
4. **run_python_sandbox** again.

If the same error happens twice, stop tweaking the same line — you likely picked the wrong API. Search docs and rewrite.

After three failed attempts on the same error, stop and ask the user for direction.
