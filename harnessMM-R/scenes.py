"""Build a tabletop layout from a JSON scene recipe and the shared tool model."""
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def compile_layout(spec):
    blocks, pads, lanes = spec["blocks"], spec["pads"], spec["source_x_m"]
    names = [*blocks, *pads]
    if (not 1 <= len(pads) <= len(blocks) == len(lanes) or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[a-z][a-z0-9_]*", n) or n in {"tool", "table", "world"} for n in names)):
        raise ValueError("use unique block/pad names and one source lane per block")
    if spec.get("occupied_pad") is not None and (spec["occupied_pad"] not in pads or len(pads) < 2):
        raise ValueError("occupied_pad requires a named pad and at least one other destination")
    if (any(len(c) != 3 or not all(0 <= v <= 1 for v in c) for c in blocks.values())
            or any(len(p) != 2 for p in pads.values())
            or not np.isfinite([*lanes, *(v for p in pads.values() for v in p)]).all()):
        raise ValueError("expected RGB triples, finite pad coordinates and finite source lanes")

    root = ET.parse(Path(__file__).with_name("tabletop.xml")).getroot()
    world, eq = root.find("worldbody"), root.find("equality")
    pad_template = world.find("body[@name='left']")
    block_template = world.find("body[@name='red']")
    weld_template = eq.find("weld")
    for item in list(world.findall("body")):
        if item.get("name") != "tool":
            world.remove(item)
    for weld in list(eq):
        eq.remove(weld)
    for i, (name, xy) in enumerate(pads.items()):
        pad = copy.deepcopy(pad_template)
        pad.set("name", name)
        pad.set("pos", f"{xy[0]} {xy[1]} .002")
        shade = .85 - .63 * i / max(1, len(pads) - 1)
        pad.find("geom").set("rgba", f"{shade} {shade} {shade+.01} 1")
        world.append(pad)
    for name, rgb in blocks.items():
        block = copy.deepcopy(block_template)
        block.set("name", name)
        block.find("geom").set("rgba", " ".join(map(str, [*rgb, 1])))
        world.append(block)
        weld = copy.deepcopy(weld_template)
        weld.attrib.update(name=name, body2=name)
        eq.append(weld)
    return ET.tostring(root, encoding="unicode")
