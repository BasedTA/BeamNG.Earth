"""Measured canopy heights for chosen areas (the real-trees patch): a canopy height model from USGS 3DEP lidar
(highest vegetation return minus the bare ground, 1 m) on the Meta / WRI model's own grid (Web Mercator, 2^25
pixels around the world: 1.194 m), laid over that model where it covers (maprasters.Rasters.local_canopy), as
LidarBC's is in British Columbia. vegetation.py then finds the trees in it as anywhere else: each real tree's top,
its height and where it stands.

Data: localcanopy_27012.npz beside this module (bundled by tools/build_app.py): one lzma-packed uint8 block of
BLOCK x BLOCK pixels per key "row_col" (block indices on the 2^25 grid), metres, 255 where the lidar area ends (the
model there is kept). VERSION changes with the data: the cached canopy and vegetation cells it covers are built again
(mapdata.MapData.path, vegetation.Vegetation._path)."""
import lzma
import os
import threading

import numpy as np

VERSION = "27012-2"          # (2: the crowns filled, vegetation._fill_canopy)
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localcanopy_27012.npz")
GRID_Z = 25                 # the pixel grid's zoom (maprasters: CHM_FILE_Z + 15)
BLOCK = 2048                # pixels a block side (one z14 map cell)
NODATA = 255
MERC_HALF = 20037508.342789244

_lock = threading.Lock()
_meta = None                # (blocks {(row, col): packed bytes}, (min row, min col, max row, max col)) or ()
_cache = {}                 # (row, col) -> decoded block (a few kept)


def _load():
    global _meta
    with _lock:
        if _meta is None:
            try:
                with np.load(PATH) as z:
                    blocks = {tuple(int(v) for v in k.split("_")): z[k].tobytes() for k in z.files}
                rs = [k[0] for k in blocks]
                cs = [k[1] for k in blocks]
                _meta = (blocks, (min(rs), min(cs), max(rs), max(cs))) if blocks else ()
            except (OSError, ValueError):
                _meta = ()
        return _meta


def _block(rc):
    with _lock:
        b = _cache.get(rc)
    if b is None:
        b = np.frombuffer(lzma.decompress(_load()[0][rc]), np.uint8).reshape(BLOCK, BLOCK)
        with _lock:
            if len(_cache) > 12:
                _cache.pop(next(iter(_cache)))
            _cache[rc] = b
    return b


def _blocks_of(cell):
    """The blocks a map cell (z <= 14) spans: (row range, col range), on the block grid."""
    s = 1 << (GRID_Z - cell.z)            # pixels a cell side
    r0, c0 = cell.y * s // BLOCK, cell.x * s // BLOCK
    n = max(1, s // BLOCK)
    return range(r0, r0 + n), range(c0, c0 + n)


def covers(cell):
    """Is any of map cell `cell`'s area measured here? (z between 10 and GRID_Z; cheap.)"""
    m = _load()
    if not m:
        return False
    if cell.z > 14:
        cell = cell.parent(14)
    rows, cols = _blocks_of(cell)
    blocks = m[0]
    return any((r, c) in blocks for r in rows for c in cols)


def window(cell):
    """The cell's measured canopy (uint8 m, NODATA outside the measured area) on the 2^GRID_Z grid, or None."""
    if not covers(cell):
        return None
    s = 1 << (GRID_Z - cell.z)
    out = np.full((s, s), NODATA, np.uint8)
    blocks = _load()[0]
    if s >= BLOCK:
        rows, cols = _blocks_of(cell)
        for r in rows:
            for c in cols:
                if (r, c) in blocks:
                    out[(r - rows.start) * BLOCK:(r - rows.start + 1) * BLOCK,
                        (c - cols.start) * BLOCK:(c - cols.start + 1) * BLOCK] = _block((r, c))
    else:                                   # a cell smaller than a block: its part of one
        p = cell.parent(14)
        rc = (p.y, p.x)
        if rc in blocks:
            r0, c0 = cell.y * s - p.y * BLOCK, cell.x * s - p.x * BLOCK
            out[:] = _block(rc)[r0:r0 + s, c0:c0 + s]
    return out if (out != NODATA).any() else None


def cell_box(cell):
    """(x0, y0 top-left in EPSG:3857, pixel size) of a cell on this grid."""
    nz = 1 << cell.z
    size = 2 * MERC_HALF / nz
    return cell.x * size - MERC_HALF, MERC_HALF - cell.y * size, 2 * MERC_HALF / (1 << GRID_Z)
