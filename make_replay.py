"""Build demo/data/replay.npz (small, float16) from the training data.
Run from the demo/ folder:  python3 make_replay.py ../start/data/patches.npz"""
import sys
from pathlib import Path
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "../start/data/patches.npz"
out = Path(__file__).parent / "data" / "replay.npz"
out.parent.mkdir(exist_ok=True)
z = np.load(src)
d = {f"sv_{t}": (z[f"sv_{t}"][:700] if t == 4 else z[f"sv_{t}"]).astype(np.float16) for t in (4, 14, 17)}
np.savez_compressed(out, **d)
print("saved", out, {k: v.shape for k, v in d.items()})
