# Source identity

P1.3 raw schema v5 adds `source.content_sha256` and `source.size_bytes` for regular local input paths. Hashing happens before model loading, warmup, and measurement. File metadata is checked around hashing and after source cleanup; a changed, replaced, deleted, or unreadable file fails the run rather than publishing a stale identity.

URLs, including `file://`, and nonregular inputs have null checksum and size. Historical missing identity remains unknown. New identity fields expose no paths. Reports continue reading older records; aggregation accepts homogeneous v3/v4/v5 groups and rejects differing source identities.

SHA-256 identifies file bytes, not decoded frames or the measured frame subset. Metadata checks detect ordinary changes; they are not an immutable-storage guarantee. Hashing reads the entire file and can warm the filesystem cache, so these runs do not establish cold-cache decode performance. Timing, FPS, and drop definitions are unchanged.
