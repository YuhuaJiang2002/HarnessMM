"""Headless sorting task with a position-controlled Cartesian suction tool."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from scenes import compile_layout


class Tabletop:
    def __init__(self, spec=None):
        if spec is None:
            spec = json.loads(Path(__file__).with_name("scenes.json").read_text())["sort-two"]
        self.spec = spec
        self.blocks, self.pads = tuple(spec["blocks"]), tuple(spec["pads"])
        self.actions = [f"pick {b}" for b in self.blocks] + [f"place {p}" for p in self.pads]
        self.model = mj.MjModel.from_xml_string(compile_layout(spec))
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, height=240, width=320)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        mj.mj_resetData(self.model, self.data)
        for name, x in zip(self.blocks, rng.permutation(self.spec["source_x_m"])):
            start = self.model.body(name).jntadr[0]
            adr = self.model.jnt_qposadr[start]
            self.data.qpos[adr:adr + 3] = [x + rng.uniform(-.025, .025),
                                            rng.uniform(-.10, -.04), .018]
        selected = rng.choice(self.blocks, len(self.pads), replace=False).tolist()
        self.goals = dict(zip(selected, rng.permutation(self.pads).tolist()))
        occupied = self.spec.get("occupied_pad")
        if occupied:
            # Start a required block on another required destination.
            block = next(b for b, p in self.goals.items() if p != occupied)
            joint = self.model.body(block).jntadr[0]
            adr = self.model.jnt_qposadr[joint]
            self.data.qpos[adr:adr + 3] = [*self.spec["pads"][occupied], .020]
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
        for pad in self.pads:
            pad_id = self.model.body(pad).geomadr[0]
            touching = ((contacts == block_id).any(axis=1) & (contacts == pad_id).any(axis=1)).any()
            if touching and np.max(np.abs(self.position(block)[:2] - self.position(pad)[:2])) < .024:
                return pad
        return None

    def observe(self):
        return {"time_s": round(self.data.time, 6), "tool_xyz_m": self.position("tool").tolist(),
                "holding": self.held, "block_xyz_m": {b: self.position(b).tolist() for b in self.blocks},
                "pad_xyz_m": {p: self.position(p).tolist() for p in self.pads},
                "resting_on": {b: self.resting_on(b) for b in self.blocks}}

    def transfers(self, observation):
        """Relate the public instruction to measured pad contacts."""
        occupant = {pad: block for block, pad in observation["resting_on"].items() if pad is not None}
        return {block: {"destination": pad, "occupied_by": occupant.get(pad)}
                for block, pad in self.goals.items()}

    def transfer_rejection(self, verb, name):
        transfers = self.transfers(self.observe())
        if verb == "pick":
            if self.held is not None:
                return f"Cannot pick {name}: the tool holds {self.held}."
            if name not in transfers:
                return f"Block {name} has no destination in this instruction."
            move = transfers[name]
            occupant = move["occupied_by"]
            if occupant == name:
                return f"Block {name} is already settled on its destination."
            if occupant is not None:
                return (f"Cannot pick {name} for {move['destination']}: {occupant} occupies it. "
                        "Clear that pad while the tool is empty.")
        elif self.held is None:
            return f"Cannot place on {name}: the tool is empty."
        elif name != transfers[self.held]["destination"]:
            return f"The assigned destination for {self.held} is {transfers[self.held]['destination']}."
        return None

    def rgb(self, camera="overhead"):
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def step(self, action, on_tick=None):
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
                              "block_xyz_m": [self.position(b).tolist() for b in self.blocks]})
                if on_tick is not None:
                    on_tick(trace[-1])

        if action not in self.actions:
            return "unknown action", trace
        verb, name = action.split()
        check_transfer = self.spec.get("occupied_pad") is not None
        if check_transfer:
            reason = self.transfer_rejection(verb, name)
            if reason is not None:
                return reason, trace
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
            if any(self.resting_on(b) == name for b in self.blocks):
                return "pad is occupied", trace
            move(above, 20)
            move(np.array([*destination[:2], .050]), 18)
            released_block = self.held
            self.data.eq_active[self.model.equality(self.held).id] = False
            self.held = None
        move(above, 20)
        if check_transfer and verb == "place" and self.resting_on(released_block) != name:
            return f"Released {released_block}, but settled contact on {name} was not observed.", trace
        return "executed", trace
