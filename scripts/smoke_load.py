"""Smoke test: does the KAIST loader load one paired sample correctly?

Run it from the repo root, inside the venv:

    python scripts/smoke_load.py

This is a *smoke test* -- the cheapest possible "is anything on fire?" check. It
doesn't prove the loader is perfect; it proves it runs and returns sane shapes.
"""
from prism.data.kaist import PairedKAIST

ds = PairedKAIST("data/raw/kaist-cvpr15-preview")
print("frames found:", len(ds))

s = ds[100]                                    # pull the 100th frame
print("id      :", s["id"])
print("rgb     :", tuple(s["rgb"].shape))      # expect (3, 512, 640)
print("thermal :", tuple(s["thermal"].shape))  # expect (3, 512, 640)
print("boxes   :", tuple(s["boxes"].shape))    # (N, 4) -- N objects in this frame
print("positives (person):", int((s["classes"] == 1).sum()))
print("ignore   (other)  :", int((s["classes"] == 0).sum()))

# Find a frame that actually has a person, so the number isn't always 0.
for i in range(len(ds)):
    if int((ds[i]["classes"] == 1).sum()) > 0:
        print(f"\nfirst frame with a person: index {i} = {ds[i]['id']}")
        break
