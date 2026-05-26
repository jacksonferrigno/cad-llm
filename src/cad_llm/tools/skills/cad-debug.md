# CAD debug (sandbox failures)

When the **sandbox run fails** (build error, traceback, or CAD script exit non-zero):

1. **`read_file`** — open the traceback paths and the files you last changed; confirm line context.
2. **`grep`** — locate symbols, imports, or duplicate definitions related to the failure.
3. **`search_replace`** — apply a minimal fix aligned with the error message.
4. **Re-run** the sandbox command that failed.

Limit this loop to **at most 3 retries** for the same failing step. After three attempts, stop, summarize what was tried, and ask for direction or a narrower hypothesis instead of spinning.
