# Manuscript build

Run the authoritative local build and validation command from this directory:

```bash
make package
```

It builds both manuscript variants, rejects fatal/undefined/overfull LaTeX
diagnostics, verifies that their core, appendix, and bibliography remain
synchronized, and refreshes the two public PDFs under `docs/`.

The wrappers intentionally differ in layout and bibliography placement:
`main.tex` is the readable draft and `main_iclr.tex` uses the conference
style. `check_sync.py` prevents their duplicated manuscript text from
drifting until they are refactored into shared includes.
