"""Headless sorting task with a position-controlled Cartesian suction tool."""
from pathlib import Path

import mujoco as mj
import numpy as np

BLOCKS = ("red", "blue", "yellow")
PADS = ("left", "right")
ACTIONS = tuple(f"pick {b}" for b in BLOCKS) + tuple(f"place {p}" for p in PADS)


class Tabletop:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(str(Path(__file__).with_name("tabletop.xml")))
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, height=240, width=320)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        mj.mj_resetData(self.model, self.data)
        for name, x in zip(BLOCKS, rng.permutation([-.13, 0., .13])):
            start = self.model.body(name).jntadr[0]
            adr = self.model.jnt_qposadr[start]
            self.data.qpos[adr:adr + 3] = [x + rng.uniform(-.025, .025),
                                            rng.uniform(-.10, -.04), .018]
        selected = rng.choice(BLOCKS, 2, replace=False).tolist()
        self.goals = dict(zip(selected, rng.permutation(PADS).tolist()))
        self.instruction = " ".join(f"Put the {b} block on the {p} pad." for b, p in self.goals.items())
        self.held = None
        mj.mj_forward(self.model, self.data)
        self.advance(100)

    def advance(self, steps):
        mj.mj_step(self.model, self.data, nstep=steps)
        mj.mj_forward(self.model, self.data)

    def position(self, name):
        return self.data.body(name).xpos.copy()

    def resting_on(self, block):
        if block == self.held:
            return None
        body = self.model.body(block)
        if np.linalg.norm(self.data.qvel[self.model.jnt_dofadr[body.jntadr[0]]:][:6]) > .02:
            return None
        block_id = body.geomadr[0]
        contacts = self.data.contact.geom
        for pad in PADS:
            pad_id = self.model.body(pad).geomadr[0]
            touching = ((contacts == block_id).any(axis=1) & (contacts == pad_id).any(axis=1)).any()
            if touching and np.max(np.abs(self.position(block)[:2] - self.position(pad)[:2])) < .024:
                return pad
        return None

    def observe(self):
        return {"time_s": round(self.data.time, 6), "tool_xyz_m": self.position("tool").tolist(),
                "holding": self.held, "block_xyz_m": {b: self.position(b).tolist() for b in BLOCKS},
                "pad_xyz_m": {p: self.position(p).tolist() for p in PADS},
                "resting_on": {b: self.resting_on(b) for b in BLOCKS}}

    def rgb(self):
        self.renderer.update_scene(self.data, camera="overhead")
        return self.renderer.render()

    def step(self, action):
        """Return execution feedback and the underlying 50 Hz control trajectory."""
        trace = []

        def move(target, ticks):
            start = self.data.mocap_pos[0].copy()
            for i in range(1, ticks + 1):
                t = i / ticks
                self.data.mocap_pos[0] = start + (target - start) * (t*t*(3 - 2*t))
                self.advance(10)
                trace.append({"time_s": round(self.data.time, 6),
                              "tool_target_xyz_m": self.data.mocap_pos[0].tolist(),
                              "suction_block": self.held,
                              "block_xyz_m": [self.position(b).tolist() for b in BLOCKS]})

        if action not in ACTIONS:
            return "unknown action", trace
        verb, name = action.split()
        destination = self.position(name)
        above = np.array([*destination[:2], .12])
        if verb == "pick":
            if self.held:
                return "tool already holds a block", trace
            move(above, 20)
            move(destination + [0, 0, .025], 18)
            if np.linalg.norm(self.position("tool") - self.position(name) - [0, 0, .025]) > .004:
                move(above, 18)
                return "pickup missed", trace
            self.data.eq_active[self.model.equality(name).id] = True
            self.held = name
        else:
            if self.held is None:
                return "tool is empty", trace
            if any(self.resting_on(b) == name for b in BLOCKS):
                return "pad is occupied", trace
            move(above, 20)
            move(np.array([*destination[:2], .050]), 18)
            self.data.eq_active[self.model.equality(self.held).id] = False
            self.held = None
        move(above, 20)
        return "executed", trace
