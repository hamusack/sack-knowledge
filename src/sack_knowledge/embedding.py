import json
import sqlite3
from contextlib import closing
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from threading import Lock
from time import perf_counter
from urllib.request import urlopen


DEFAULT_MODEL = "cl-nagoya/ruri-v3-30m"
EXPORT = "https://huggingface.co/kens24481/ruri-v3-30m-onnx/resolve/37cdd04ec40dc9e691b53011d4f53b5e17ecab99/"


def allowed_chunk(alias="c"):
    return f"""EXISTS (
        SELECT 1 FROM records anchor
        WHERE anchor.id={alias}.first_record_id AND anchor.deleted=0
          AND (anchor.source <> 'discord' OR EXISTS (
              SELECT 1 FROM allow_list a
              WHERE a.source=anchor.source AND a.target=anchor.account AND a.local_storage=1
          ))
    )"""


def embedding_name(config):
    return config.get("embedding", {}).get("model", DEFAULT_MODEL)


def vector_key(config):
    options = config.get("embedding", {})
    # Changes that alter inference invalidate stored vectors even when the name stays the same.
    return json.dumps([embedding_name(config), options.get("max_tokens", 256),
                       options.get("query_prefix", "検索クエリ: "),
                       options.get("passage_prefix", "検索文書: "),
                       options.get("model_path"), options.get("tokenizer_path")], ensure_ascii=False)


def download(path, url):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            with urlopen(url, timeout=60) as response:
                copyfileobj(response, file)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class LocalEmbedder:
    def __init__(self, config, *, allow_download=False):
        self.options = config.get("embedding", {})
        self.name = embedding_name(config)
        self.allow_download = allow_download
        self.lock = Lock()
        self.session = None

    def load(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        ort.disable_telemetry_events()
        options = self.options
        cache = Path(options.get("cache", "local/models")) / "ruri-v3-30m"
        if self.name != DEFAULT_MODEL or "model_path" in options or "tokenizer_path" in options:
            model = Path(options["model_path"])
            tokenizer = Path(options["tokenizer_path"])
        else:
            model, tokenizer = cache / "model.onnx", cache / "tokenizer.json"
            if self.allow_download:
                download(model, EXPORT + "onnx/model.onnx")
                download(tokenizer, EXPORT + "tokenizer.json")
        tokens = Tokenizer.from_file(str(tokenizer))
        tokens.enable_truncation(max_length=options.get("max_tokens", 256))
        pad = tokens.token_to_id("[PAD]")
        tokens.enable_padding(pad_id=pad if pad is not None else 0)
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = options.get("threads", 8)
        settings.log_severity_level = 4
        session = ort.InferenceSession(str(model), sess_options=settings, providers=["CPUExecutionProvider"])
        self.tokenizer = tokens
        self.session = session

    def encode(self, texts, *, query=False):
        import numpy as np

        with self.lock:
            if self.session is None:
                self.load()
        prefix = (self.options.get("query_prefix", "検索クエリ: ") if query
                  else self.options.get("passage_prefix", "検索文書: "))
        encoded = self.tokenizer.encode_batch([prefix + text for text in texts])
        inputs = {
            "input_ids": np.array([item.ids for item in encoded], dtype=np.int64),
            "attention_mask": np.array([item.attention_mask for item in encoded], dtype=np.int64),
        }
        token_vectors = self.session.run(None, inputs)[0]
        mask = inputs["attention_mask"][:, :, None].astype(np.float32)
        return (token_vectors * mask).sum(axis=1) / mask.sum(axis=1)


def unit_vectors(matrix):
    import numpy as np

    matrix = np.asarray(matrix, dtype="<f4")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("Invalid embedding norm")
    return matrix / norms[:, None]


def embed(config, embedder=None):
    started = perf_counter()
    path = Path(config["storage"]["database"])
    name, key = embedding_name(config), vector_key(config)
    batch_size = config.get("embedding", {}).get("batch_size", 32)
    if batch_size < 1:
        raise ValueError("Invalid embedding batch size")
    embedder = embedder if embedder is not None else LocalEmbedder(config, allow_download=True)
    embedded = 0
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=60)) as db:
        with db:
            db.execute("BEGIN IMMEDIATE")
            if not any(row[1] == "embedding_dimension" for row in db.execute("PRAGMA table_info(chunks)")):
                db.execute("ALTER TABLE chunks ADD COLUMN embedding_dimension INTEGER")
            if not any(row[1] == "embedding_key" for row in db.execute("PRAGMA table_info(chunks)")):
                db.execute("ALTER TABLE chunks ADD COLUMN embedding_key TEXT")
        # Length sorting reduces padding, and chunk IDs change on sync edits.
        rows = db.execute(f"""
            SELECT c.id, c.body FROM chunks c
            WHERE (c.embedding IS NULL OR c.embedding_key IS NOT ?) AND {allowed_chunk()}
            ORDER BY length(c.body), c.id
        """, (key,)).fetchall()
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            vectors = unit_vectors(embedder.encode([body for _, body in batch]))
            with db:
                cursor = db.executemany(f"""
                    UPDATE chunks AS c SET embedding=?, embedding_model=?, embedding_dimension=?, embedding_key=?
                    WHERE c.id=? AND (c.embedding IS NULL OR c.embedding_key IS NOT ?) AND {allowed_chunk()}
                """, ((vector.tobytes(), name, len(vector), key, identifier, key)
                      for (identifier, _), vector in zip(batch, vectors, strict=True)))
                embedded += cursor.rowcount
        dimension = db.execute("""
            SELECT embedding_dimension FROM chunks WHERE embedding_key=? AND embedding IS NOT NULL LIMIT 1
        """, (key,)).fetchone()
    return {"embedded": embedded, "model": name, "dimension": dimension[0] if dimension else None,
            "elapsed_ms": round((perf_counter() - started) * 1000, 2), "sqlite_bytes": path.stat().st_size}
