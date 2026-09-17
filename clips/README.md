# Sample clips

Video files are not committed. Fetch them with:

```bash
python scripts/fetch_clips.py
```

The manifest is [`sources.json`](sources.json). Only redistributable,
permissively licensed footage belongs in it.

Two rules for anything added here:

1. **No footage from a live deployment.** People recorded by a security camera
   did not consent to appearing in a public repository, and in most
   jurisdictions that footage is personal data. Benchmark against real streams
   locally, and publish only the numbers.
2. **Record the licence and attribution** in the manifest, not just the URL.

For benchmarking against your own cameras, pass the stream URL directly. It
never touches the repository, and credentials are stripped from results before
they are written to disk.

```bash
visionbench run -s "rtsp://user:pass@host:554/stream" -m dfine-nano -d 60
```
