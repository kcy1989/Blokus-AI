"""This layer answers one question: what does a training batch look like, given
a row?

Three things have to be true at once for a batch to be usable, and this module
owns all three:

  * the features. `featurize` gives 14 planes and 13 scalars; H-B2 decision 17
    feeds the scalars in as 13 broadcast channels, so the network takes 27.
  * the legal-action mask. `legal_mask_view` is 152 ms per 256 rows and the GPU
    step is 42 ms, so the mask is precomputed once per row and stored sparse -
    240,569,093 indices, restored in 1.9 ms.
  * the label, in the mover's view frame. The network's head is in the view, so
    a real action index is useless as a target until it is mapped across. Also
    precomputed.

Both caches are indexed by **global row order** - every row of every shard in
`shard_order()`, train and valid concatenated. There is deliberately no
split-local index anywhere in this module: an earlier smoke test indexed the
caches by position within a split while taking the features from a different
array, and the validation loss came out at 954,099,765. One index drives all
three here, and `assert_targets_legal` checks it.

Nothing in this module imports torch, so a DataLoader worker can call it without
touching CUDA.
"""
import json
import os

import numpy as np

from config import CLOCKWISE_OWNERS
from rl.actions import legal_mask_view, real_to_view
from rl.collect import load_shard
from rl.features import featurize_batch

N_ACTIONS = 91 * 20 * 20
LEGAL_CACHE_NAME = "legalactions.npz"
LABEL_CACHE_NAME = "labels_view.npy"

# the columns a training loop actually reads; everything else in the shard is
# for statistics or for H-B4
COLUMNS = ("own_bits", "hand_bits", "stuck", "to_move", "own_move_count")


def shard_order(data_dir):
    """Shards in the order both caches were built in."""
    return sorted(f for f in os.listdir(data_dir)
                  if f.startswith("hb1_") and f.endswith(".npz"))


def split_offsets(data_dir, names=None):
    """Where each split starts in the global row order."""
    names = names or shard_order(data_dir)
    offsets = {}
    row = 0
    for which in ("train", "valid"):
        offsets[which] = row
        for n in names:
            if which in n:
                row += _shard_rows(data_dir, n)
    return offsets, row


def _shard_rows(data_dir, name):
    return int(load_shard(os.path.join(data_dir, name))["action"].size)


# --------------------------------------------------------------- mmap access

def _npz_member(path, member, dtype):
    """Memory-map one array out of an uncompressed `.npz`.

    `np.savez` writes each array as a `.npy` member with `ZIP_STORED`, so the
    payload sits in the file verbatim and can be mapped once two headers are
    skipped: the zip local header, and the `.npy` magic/shape/dtype header
    inside the member. The shape comes from that header rather than being
    assumed, so a wrong offset cannot quietly produce a differently-sized array
    - `np.memmap` is given the shape explicitly and fails if the file does not
    hold it.

    A compressed member has no mappable bytes at all: they do not exist until the
    member is inflated. That is raised, not worked around, because the whole
    point of mapping is that six DataLoader workers must not each copy 931 MiB.
    """
    import zipfile

    from numpy.lib import format as npformat

    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                "%s stores %r with compression method %d; a compressed member "
                "has no mappable bytes, so every process would have to read "
                "and inflate it whole. Rebuild it with np.savez (uncompressed)"
                % (os.path.basename(path), member, info.compress_type))
        with open(path, "rb") as fh:
            fh.seek(info.header_offset)
            head = fh.read(30)
            name_len = int.from_bytes(head[26:28], "little")
            extra_len = int.from_bytes(head[28:30], "little")
            fh.seek(info.header_offset + 30 + name_len + extra_len)
            version = npformat.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype_ = npformat.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran, dtype_ = npformat.read_array_header_2_0(fh)
            else:
                raise ValueError("unsupported .npy version %r in %s"
                                 % (version, member))
            data_at = fh.tell()
    if dtype_ != np.dtype(dtype):
        raise ValueError("%s stores %r as %s, expected %s"
                         % (os.path.basename(path), member, dtype_, dtype))
    if fortran:
        raise ValueError("%s stores %r in Fortran order, which cannot be mapped "
                         "as C order" % (os.path.basename(path), member))
    return data_at, shape


def mmap_npz_array(path, member, dtype):
    off, shape = _npz_member(path, member, dtype)
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=tuple(shape),
                     order="C")


def open_legal_cache(data_dir, mmap=True):
    """`(flat, offsets)` for the sparse legal-action cache."""
    path = os.path.join(data_dir, LEGAL_CACHE_NAME)
    if mmap:
        try:
            flat = mmap_npz_array(path, "flat", np.int32)
            offsets = mmap_npz_array(path, "offsets", np.int64)
            return flat, offsets
        except ValueError as exc:
            raise ValueError(
                "%s cannot be memory-mapped: %s" % (LEGAL_CACHE_NAME, exc))
    z = np.load(path)
    return z["flat"], z["offsets"]


def open_label_cache(data_dir, mmap=True):
    """One int32 per row: the action label in the mover's view frame."""
    path = os.path.join(data_dir, LABEL_CACHE_NAME)
    return np.load(path, mmap_mode="r" if mmap else None)


# ------------------------------------------------------------------ sources

class ShardArrays:
    """Lazy, per-process view of the shards.

    Used by DataLoader workers, which each open this once and mmap rather than
    copying: the legal cache alone is 931 MiB, and six workers copying it would
    cost 5.5 GiB for nothing.
    """

    def __init__(self, data_dir, mmap=True):
        self.data_dir = data_dir
        self.names = shard_order(data_dir)
        self.offsets, self.n_rows = split_offsets(data_dir, self.names)
        self._mmap = mmap
        self._shards = {}
        self.legal_flat, self.legal_off = open_legal_cache(data_dir, mmap)
        self.labels = open_label_cache(data_dir, mmap)

    def shard(self, name):
        """Open one shard on first use; a spawn worker builds its own."""
        d = self._shards.get(name)
        if d is None:
            path = os.path.join(self.data_dir, name)
            d = np.load(path) if self._mmap else load_shard(path)
            if self._mmap:
                # keep only the columns this module needs resident
                d = {c: d[c] for c in COLUMNS}
            self._shards[name] = d
        return d

    def locate(self, gidx):
        """Global row -> (shard name, local row)."""
        for n in self.names:
            k = self._shard_of(n)
            if gidx < k:
                return n, gidx
            gidx -= k
        raise IndexError("global row out of range")

    def _shard_of(self, name):
        cached = getattr(self, "_rows_cache", None)
        if cached is None:
            cached = self._rows_cache = {}
        if name not in cached:
            d = np.load(os.path.join(self.data_dir, name))
            cached[name] = int(d["action"].size)
        return cached[name]

    def states(self, gidx):
        return states_from_rows(self.gather(gidx))

    def mask_for(self, gidx):
        return mask_from_cache(self.legal_flat, self.legal_off, gidx)

    def labels_for(self, gidx):
        return np.asarray(self.labels[gidx], dtype=np.int64)

    def gather(self, gidx):
        """Rows by global index, as a dict of arrays aligned on the batch."""
        out = {c: [] for c in COLUMNS}
        for g in gidx:
            name, local = self.locate(int(g))
            d = self.shard(name)
            for c in COLUMNS:
                out[c].append(d[c][local])
        return {c: np.array(v) for c, v in out.items()}


class Resident:
    """Every needed column concatenated in RAM, in global row order.

    The serial loop's source. Same interface as `ShardArrays`, so `make_batch`
    does not know which one it is talking to.
    """

    def __init__(self, data_dir, mmap=False):
        self.data_dir = data_dir
        self.names = shard_order(data_dir)
        self.offsets, self.n_rows = split_offsets(data_dir, self.names)
        cols = {c: [] for c in COLUMNS}
        for n in self.names:
            d = load_shard(os.path.join(data_dir, n))
            for c in COLUMNS:
                cols[c].append(d[c])
        self.col = {c: np.concatenate(cols[c]) for c in COLUMNS}
        self.legal_flat, self.legal_off = open_legal_cache(data_dir, mmap)
        self.labels = open_label_cache(data_dir, mmap)

    def states(self, gidx):
        return states_from_rows(self.gather(gidx))

    def mask_for(self, gidx):
        return mask_from_cache(self.legal_flat, self.legal_off, gidx)

    def labels_for(self, gidx):
        return np.asarray(self.labels[gidx], dtype=np.int64)

    def gather(self, gidx):
        g = np.asarray(gidx, dtype=np.int64)
        return {c: self.col[c][g] for c in COLUMNS}


# ------------------------------------------------------------------- batch

def states_from_rows(rows):
    """Shard columns -> `engine.State` objects.

    A namedtuple would be faster, but `featurize` and `legal_mask_view` both
    want a real `engine.State`, and building one per row is 3.4 ms per 256.
    """
    import engine

    out = []
    for i in range(len(rows["own_bits"])):
        gb = rows["own_bits"][i]
        hb = rows["hand_bits"][i]
        sb = rows["stuck"][i]
        out.append(engine.State(
            own_bits=tuple(int.from_bytes(bytes(gb[o]), "little")
                           for o in range(4)),
            hand_bits=tuple(int(hb[o]) for o in range(4)),
            turn_order=CLOCKWISE_OWNERS,
            to_move=int(rows["to_move"][i]),
            stuck=tuple(bool(sb[o]) for o in range(4))))
    return out


def features_27(states):
    """`(N, 27, 20, 20)` float32: 14 planes then the 13 scalars broadcast."""
    planes, scalars, _hands = featurize_batch(states)
    extra = np.repeat(scalars[:, :, None, None], planes.shape[2], axis=2)
    extra = np.repeat(extra, planes.shape[3], axis=3)
    return np.ascontiguousarray(np.concatenate([planes, extra], axis=1))


def mask_from_cache(flat, offsets, gidx):
    """Sparse legal indices -> a `(len(gidx), 36400)` bool mask."""
    m = np.zeros((len(gidx), N_ACTIONS), dtype=bool)
    for r, g in enumerate(gidx):
        g = int(g)
        m[r, flat[offsets[g]:offsets[g + 1]]] = True
    return m


def make_batch(source, gidx):
    """Global row indices -> `(x27, mask, labels)`, all numpy.

    The single entry point both training loops use, so the DataLoader version
    cannot drift from the serial one.
    """
    gidx = [int(g) for g in gidx]
    rows = source.gather(gidx)
    states = states_from_rows(rows)
    x = features_27(states)
    mask = mask_from_cache(source.legal_flat, source.legal_off, gidx)
    labels = np.asarray(source.labels[gidx], dtype=np.int64)
    return x, mask, labels


def assert_targets_legal(mask, labels, where, gidx=None):
    """Every target must be one of that row's own legal slots.

    A gather and an `all`, so it is cheap enough to sit in the hot loop. A
    failure means the caches and the rows are indexed differently, which is the
    defect that once made the validation loss read 954,099,765.
    """
    hit = mask[np.arange(len(labels)), labels]
    bad = np.flatnonzero(~hit)
    if bad.size:
        extra = ""
        if gidx is not None:
            extra = "; first global rows %s" % [int(gidx[b]) for b in bad[:5]]
        raise AssertionError(
            "%s: %d of %d targets are outside their own legal mask%s"
            % (where, bad.size, len(labels), extra))


# ------------------------------------------------------------------ builders

def build_legal_cache(data_dir, out_name=LEGAL_CACHE_NAME, workers=12):
    """Precompute every row's legal-action indices. Data preprocessing."""
    import multiprocessing as mp

    names = shard_order(data_dir)
    chunks = []
    for n in names:
        k = _shard_rows(data_dir, n)
        step = max(1, -(-k // workers))
        for lo in range(0, k, step):
            chunks.append((os.path.join(data_dir, n), lo, min(lo + step, k)))

    def one(args):
        path, lo, hi = args
        d = load_shard(path)
        states = states_from_rows({c: d[c][lo:hi] for c in COLUMNS})
        parts = []
        counts = np.empty(hi - lo, dtype=np.int32)
        for k, st in enumerate(states):
            idx = np.flatnonzero(legal_mask_view(st)).astype(np.int32)
            parts.append(idx)
            counts[k] = idx.size
        return path, lo, np.concatenate(parts) if parts else \
            np.empty(0, np.int32), counts

    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(one, chunks)
    order = sorted(range(len(parts)), key=lambda i: (parts[i][0], parts[i][1]))
    flat = np.concatenate([parts[i][1] for i in order])
    counts = np.concatenate([parts[i][2] for i in order])
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    out = os.path.join(data_dir, out_name)
    if os.path.exists(out):
        raise FileExistsError("%s exists; refusing to overwrite" % out)
    np.savez(out, flat=flat, offsets=offsets, counts=counts,
             shards=np.array(names), rows=int(counts.size))
    return {"rows": int(counts.size), "indices": int(flat.size),
            "bytes": os.path.getsize(out), "path": out}


def build_label_cache(data_dir, out_name=LABEL_CACHE_NAME, workers=12):
    """Precompute every row's view-frame label. Data preprocessing.

    Both inputs are already shard columns, so nothing is rebuilt, and
    `real_to_view` is one fancy index per seat group per shard rather than one
    call per row.
    """
    import multiprocessing as mp

    names = shard_order(data_dir)
    chunks = []
    for n in names:
        k = _shard_rows(data_dir, n)
        step = max(1, -(-k // workers))
        for lo in range(0, k, step):
            chunks.append((os.path.join(data_dir, n), lo, min(lo + step, k)))

    def one(args):
        path, lo, hi = args
        d = load_shard(path)
        action = d["action"][lo:hi].astype(np.int64)
        to_move = d["to_move"][lo:hi].astype(np.int64)
        out = np.empty(hi - lo, dtype=np.int32)
        for owner in range(4):
            p = CLOCKWISE_OWNERS.index(owner)
            sel = np.flatnonzero(to_move == owner)
            if sel.size:
                out[sel] = real_to_view(action[sel], p)
        return path, lo, out

    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(one, chunks)
    order = sorted(range(len(parts)), key=lambda i: (parts[i][0], parts[i][1]))
    labels = np.concatenate([parts[i][2] for i in order])
    out = os.path.join(data_dir, out_name)
    if os.path.exists(out):
        raise FileExistsError("%s exists; refusing to overwrite" % out)
    np.save(out, labels)
    return {"rows": int(labels.size), "bytes": os.path.getsize(out),
            "path": out}


def read_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)