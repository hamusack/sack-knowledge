import json
import sqlite3
from contextlib import closing
from functools import lru_cache
from hashlib import file_digest, sha256
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from threading import Lock
from time import perf_counter
from urllib.request import urlopen

from .errors import LocalCommandError, ModelIntegrityError
from .visibility import visible_chunk


DEFAULT_MODEL = "cl-nagoya/ruri-v3-30m"
EXPORT = "https://huggingface.co/kens24481/ruri-v3-30m-onnx/resolve/37cdd04ec40dc9e691b53011d4f53b5e17ecab99/"
MODEL_HASH = "96150feeedd0bddeb6d23e601814288ff362a3b62e93d04caf17d0d0432481e5"
TOKENIZER_HASH = "0a94ac9a0a02c067bdef25b72ae9f4ee33f48f552e55988d444f6d25eeb1d062"
MAX_TOKENS = 8192
QUERY_PREFIX = "検索クエリ: "
PASSAGE_PREFIX = "検索文書: "


def embedding_name(config):
    return config.get("embedding", {}).get("model", DEFAULT_MODEL)


def model_files(config):
    options = config.get("embedding", {})
    pinned = (MODEL_HASH, TOKENIZER_HASH) if embedding_name(config) == DEFAULT_MODEL else None
    if pinned is None or "model_path" in options or "tokenizer_path" in options:
        if not options.get("model_path") or not options.get("tokenizer_path"):
            raise LocalCommandError("Configure both the ONNX and tokenizer files for a custom model.")
        return Path(options["model_path"]), Path(options["tokenizer_path"]), pinned
    cache = Path(options.get("cache", "local/models")) / "ruri-v3-30m"
    return cache / "model.onnx", cache / "tokenizer.json", pinned


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=16)
def cached_hash(path, stamp):
    with path.open("rb") as file:
        digest = file_digest(file, "sha256").hexdigest()
    if signature(path) != stamp:
        raise LocalCommandError("An embedding asset changed while being verified.")
    return digest


def file_hash(path):
    return cached_hash(path.resolve(), signature(path))


def verify_hash(actual, expected):
    if actual != expected:
        raise ModelIntegrityError()


def vector_key(config):
    options = config.get("embedding", {})
    model, tokenizer, pinned = model_files(config)
    if pinned:
        for path, digest in zip((model, tokenizer), pinned, strict=True):
            if path.exists():
                verify_hash(file_hash(path), digest)
        hashes = pinned
    else:
        try:
            hashes = file_hash(model), file_hash(tokenizer)
        except FileNotFoundError:
            raise LocalCommandError("The configured ONNX or tokenizer file is missing.") from None
    limit = options.get("max_tokens", MAX_TOKENS)
    if type(limit) is not int or limit < 1 or (pinned and limit > MAX_TOKENS):
        raise LocalCommandError("Invalid embedding token limit.")
    return json.dumps([embedding_name(config), *hashes, limit,
                       options.get("query_prefix", QUERY_PREFIX),
                       options.get("passage_prefix", PASSAGE_PREFIX)], ensure_ascii=False)


def download(path, url, expected):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            with urlopen(url, timeout=60) as response:
                copyfileobj(response, file)
            file.close()
            verify_hash(file_hash(temporary), expected)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_default_files(config):
    model, tokenizer, pinned = model_files(config)
    if pinned:
        download(model, EXPORT + "onnx/model.onnx", pinned[0])
        download(tokenizer, EXPORT + "tokenizer.json", pinned[1])


class LocalEmbedder:
    def __init__(self, config, *, allow_download=False):
        self.config = config
        self.options = config.get("embedding", {})
        self.allow_download = allow_download
        self.lock = Lock()
        self.session = None
        self.key = None

    def load(self, key):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        ort.disable_telemetry_events()
        model, tokenizer, pinned = model_files(self.config)
        _, model_hash, tokenizer_hash, _, _, _ = json.loads(key)
        if pinned and self.allow_download:
            prepare_default_files(self.config)
        try:
            model_data, tokenizer_data = model.read_bytes(), tokenizer.read_bytes()
        except FileNotFoundError:
            raise LocalCommandError("Embedding model files are missing. Run embed to prepare the cache.") from None
        verify_hash(sha256(model_data).hexdigest(), model_hash)
        verify_hash(sha256(tokenizer_data).hexdigest(), tokenizer_hash)
        tokens = Tokenizer.from_buffer(tokenizer_data)
        tokens.no_truncation()
        tokens.no_padding()
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = self.options.get("threads", 8)
        settings.log_severity_level = 4
        session = ort.InferenceSession(model_data, sess_options=settings, providers=["CPUExecutionProvider"])
        self.tokenizer = tokens
        self.session = session
        self.key = key

    def runtime(self, key):
        with self.lock:
            if self.key != key:
                self.load(key)
            return self.tokenizer, self.session

    def lengths(self, texts, *, key):
        tokenizer, _ = self.runtime(key)
        prefix = json.loads(key)[-1]
        return [len(item.ids) for item in tokenizer.encode_batch([prefix + text for text in texts])]

    def encode(self, texts, *, key, query=False):
        import numpy as np

        tokenizer, session = self.runtime(key)
        _, _, _, limit, query_prefix, passage_prefix = json.loads(key)
        prefix = query_prefix if query else passage_prefix
        encoded = tokenizer.encode_batch([prefix + text for text in texts])
        windows = [(index, item.ids[start:start + limit]) for index, item in enumerate(encoded)
                   for start in range(0, len(item.ids), limit)]
        windows.sort(key=lambda window: len(window[1]))
        if not windows or any(not item.ids for item in encoded):
            raise LocalCommandError("The embedding tokenizer produced an empty sequence.")
        pad = tokenizer.token_to_id("[PAD]") or 0
        totals = None
        start = 0
        while start < len(windows):
            end = start + 1
            while end < len(windows) and len(windows[end][1]) * (end - start + 1) <= MAX_TOKENS:
                end += 1
            batch = windows[start:end]
            width = len(batch[-1][1])
            ids = np.full((len(batch), width), pad, dtype=np.int64)
            attention = np.zeros_like(ids)
            for row, (_, tokens) in enumerate(batch):
                ids[row, :len(tokens)] = tokens
                attention[row, :len(tokens)] = 1
            values = session.run(None, {"input_ids": ids, "attention_mask": attention})[0]
            sums = (values * attention[:, :, None].astype(np.float32)).sum(axis=1)
            if totals is None:
                totals = np.zeros((len(encoded), sums.shape[1]), dtype=np.float32)
            for (index, _), value in zip(batch, sums, strict=True):
                totals[index] += value
            start = end
        return totals / np.array([len(item.ids) for item in encoded], dtype=np.float32)[:, None]


def unit_vectors(matrix):
    import numpy as np

    matrix = np.asarray(matrix, dtype="<f4")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("Invalid embedding norm")
    return matrix / norms[:, None]


def awaiting_embeddings(db, key):
    predicate = "1"
    params = ()
    if key is not None and any(row[1] == "embedding_key" for row in db.execute("PRAGMA table_info(chunks)")):
        predicate = "(c.embedding IS NULL OR c.embedding_key IS NOT ?)"
        params = (key,)
    return db.execute(f"SELECT count(*) FROM chunks c WHERE {visible_chunk()} AND {predicate}", params).fetchone()[0]


def embed(config, embedder=None):
    started = perf_counter()
    path = Path(config["storage"]["database"])
    name, key = embedding_name(config), vector_key(config)
    batch_size = config.get("embedding", {}).get("batch_size", 32)
    if batch_size < 1:
        raise ValueError("Invalid embedding batch size")
    prepare_files = embedder is None
    if prepare_files:
        embedder = LocalEmbedder(config, allow_download=True)
    embedded = 0
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=60)) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")
            if not any(row[1] == "embedding_dimension" for row in db.execute("PRAGMA table_info(chunks)")):
                db.execute("ALTER TABLE chunks ADD COLUMN embedding_dimension INTEGER")
            if not any(row[1] == "embedding_key" for row in db.execute("PRAGMA table_info(chunks)")):
                db.execute("ALTER TABLE chunks ADD COLUMN embedding_key TEXT")
        if prepare_files and db.execute(f"SELECT 1 FROM chunks c WHERE {visible_chunk()} LIMIT 1").fetchone():
            prepare_default_files(config)
        # Chunk IDs change on sync edits. Token-length sorting limits padding.
        rows = db.execute(f"""
            SELECT c.id, c.body FROM chunks c
            WHERE (c.embedding IS NULL OR c.embedding_key IS NOT ?) AND {visible_chunk()}
            ORDER BY c.id
        """, (key,)).fetchall()
        lengths = []
        for start in range(0, len(rows), 512):
            lengths.extend(embedder.lengths([body for _, body in rows[start:start + 512]], key=key))
        rows = [row for _, row in sorted(zip(lengths, rows, strict=True), key=lambda pair: pair[0])]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            vectors = unit_vectors(embedder.encode([body for _, body in batch], key=key))
            with db:
                cursor = db.executemany(f"""
                    UPDATE chunks AS c SET embedding=?, embedding_model=?, embedding_dimension=?, embedding_key=?
                    WHERE c.id=? AND (c.embedding IS NULL OR c.embedding_key IS NOT ?) AND {visible_chunk()}
                """, ((vector.tobytes(), name, len(vector), key, identifier, key)
                      for (identifier, _), vector in zip(batch, vectors, strict=True)))
                embedded += cursor.rowcount
        dimension = db.execute("""
            SELECT embedding_dimension FROM chunks WHERE embedding_key=? AND embedding IS NOT NULL LIMIT 1
        """, (key,)).fetchone()
    return {"embedded": embedded, "model": name, "dimension": dimension[0] if dimension else None,
            "elapsed_ms": round((perf_counter() - started) * 1000, 2), "sqlite_bytes": path.stat().st_size}
