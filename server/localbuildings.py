"""Measured buildings for chosen areas (the real-buildings patch): per Overture building id, the attributes that
plan() otherwise predicts, measured from USGS 3DEP lidar (height, roof shape, roof height and pitch) and NAIP aerial
photos (roof colour). Merged over Overture's own attributes in buildings.collect (only the fields measured; Overture's
mapped values win where it has them, except height, which lidar measures better than the ML estimates).

The data is a module (localbuildings_data.py: a gzipped JSON blob), so the packaged .exe carries it and any change to
it changes revisit.code_key, which rebuilds cached tiles."""
import base64
import gzip
import json

_DATA = None


def _load():
    global _DATA
    if _DATA is None:
        try:
            import localbuildings_data
            _DATA = json.loads(gzip.decompress(base64.b64decode(localbuildings_data.BLOB)))
        except Exception:
            _DATA = {}
    return _DATA


def get(bid):
    """The measured attributes of Overture building `bid` (dict), or None."""
    if bid is None:
        return None
    return _load().get(str(bid))


def merge(at, bid):
    """Merge building `bid`'s measured attributes into plan()'s attribute dict `at` (in place)."""
    m = get(bid)
    if not m:
        return at
    if m.get("height"):
        at["height"] = float(m["height"])
        at["num_floors"] = None              # (the floors follow the measured walls)
    if not str(at.get("roof_shape") or "") and m.get("roof_shape"):
        at["roof_shape"] = m["roof_shape"]
    if m.get("roof_height") is not None and (at.get("roof_height") is None or not at["roof_height"] == at["roof_height"]):
        at["roof_height"] = float(m["roof_height"])
    if m.get("roof_pitch"):
        at["roof_pitch"] = float(m["roof_pitch"])
    if not at.get("roof_color") and m.get("roof_color"):
        at["roof_color"] = m["roof_color"]
    if not at.get("facade_color") and m.get("facade_color"):
        at["facade_color"] = m["facade_color"]
    return at
