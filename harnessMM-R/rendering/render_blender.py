"""Render synchronized tabletop observations from recorded rigid poses with Blender Cycles."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import bpy
from mathutils import Matrix

ASSETS = Path(__file__).resolve().with_name("assets")


def material(name, color, metal=0, rough=.4, micro=False):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    nodes, links = m.node_tree.nodes, m.node_tree.links
    shader = nodes.get("Principled BSDF")
    for prop, value in (("Base Color", (*color, 1)), ("Metallic", metal), ("Roughness", rough)):
        shader.inputs[prop].default_value = value
    if micro:
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 330
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = .00005
        bump.inputs["Strength"].default_value = .18
        links.new(tex.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return m


def timber():
    m = material("Scanned workbench hardwood", (.32,.18,.08))
    nodes, links = m.node_tree.nodes, m.node_tree.links
    uv = nodes.new("ShaderNodeTexCoord")
    shader = nodes.get("Principled BSDF")
    for filename, socket in (("diff.jpg","Base Color"), ("rough.jpg","Roughness"), ("nor_gl.jpg","Normal")):
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(ASSETS / filename))
        tex.image.colorspace_settings.name = "sRGB" if socket=="Base Color" else "Non-Color"
        links.new(uv.outputs["Generated"], tex.inputs["Vector"])
        output = tex.outputs["Color"]
        if socket=="Roughness":
            remap=nodes.new("ShaderNodeMapRange")
            remap.inputs["To Min"].default_value=.43
            remap.inputs["To Max"].default_value=.72
            links.new(output,remap.inputs["Value"])
            output=remap.outputs["Result"]
        if socket=="Normal":
            normal = nodes.new("ShaderNodeNormalMap")
            normal.inputs["Strength"].default_value = .36
            links.new(output, normal.inputs["Color"])
            output = normal.outputs["Normal"]
        links.new(output, shader.inputs[socket])
    return m


def part(name, pos, mat, parent=None, bevel=0):
    o = bpy.context.object
    o.name, o.parent, o.location = name, parent, pos
    o.data.materials.append(mat)
    if bevel:
        b = o.modifiers.new("Machined edges", "BEVEL")
        b.width, b.segments = bevel, 3
        n = o.modifiers.new("Face normals", "WEIGHTED_NORMAL")
        n.keep_sharp = True
    return o


def box(name, pos, dims, mat, parent=None, bevel=.0006):
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object
    o.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return part(name, pos, mat, parent, bevel)


def cylinder(name, pos, radius, depth, mat, parent=None, vertices=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth)
    return part(name, pos, mat, parent, min(.00024, depth/5))


def empty(name):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    return obj


def screw(x,y,z,parent=None, scale=1):
    cylinder("Socket screw", (x,y,z), .0025*scale,.0006*scale,steel,parent)
    cylinder("Recess", (x,y,z+.00031*scale), .0012*scale,.00008*scale,black,parent,6)


def lettering(value, pos, size, mat, parent=None):
    data=bpy.data.curves.new("Marking","FONT")
    data.body, data.size = value, size
    o=bpy.data.objects.new("Mark "+value,data)
    bpy.context.collection.objects.link(o)
    o.parent, o.location = parent, pos
    data.materials.append(mat)


def hose(points, radius, mat, parent):
    c=bpy.data.curves.new("Air line","CURVE")
    c.dimensions="3D"
    c.resolution_u, c.bevel_depth, c.bevel_resolution = 12, radius, 3
    sp=c.splines.new("BEZIER")
    sp.bezier_points.add(len(points)-1)
    for p,xyz in zip(sp.bezier_points,points):
        p.co=xyz
        p.handle_left_type=p.handle_right_type="AUTO"
    o=bpy.data.objects.new("Pneumatic tube",c)
    bpy.context.collection.objects.link(o)
    o.parent=parent
    c.materials.append(mat)


def build(task):
    recipe=task["recipe"]
    box("Workbench", (0,0,-.022), (.84,.58,.044),timber(),bevel=.004)
    # The ruler and the fasteners are surface dressing outside the action area.
    box("Steel rule", (0,-.193,.0006), (.47,.016,.0012),steel)
    for i in range(91):
        x=-.225+i*.005
        length=.008 if i%10==0 else (.005 if i%2==0 else .003)
        box("Etched graduation",(x,-.185-length/2,.00123),(.00027,length,.00004),black,bevel=0)
        if i%10==0:
            lettering(str(i*5),(x+.001,-.199,.00126),.004,black)
    for i,(name,xy) in enumerate(recipe["pads"].items()):
        shade=.64-.48*i/max(1,len(recipe["pads"])-1)
        mat=material("Fixture "+name,(shade,shade*.98,shade*.93),.75,.36,True)
        x,y=xy
        box("Fixture "+name,(x,y,.002),(.086,.086,.004),mat,bevel=.0013)
        for sx in (-1,1):
            for sy in (-1,1): screw(x+sx*.0355,y+sy*.0355,.004)
        lettering(name.upper(),(x-.024,y-.036,.00405),.0048,black)
        for sign in (-1,1):
            box("Datum",(x+sign*.027,y,.00403),(.005,.0004,.00003),black,bevel=0)
    bodies={}
    for i,(name,rgb) in enumerate(recipe["blocks"].items()):
        root=empty("Body "+name)
        bodies[name]=root
        # Milled, pigmented polymer with a fine edge break and a batch mark.
        rgb=tuple(max(.012,v**1.65) for v in rgb)
        m=material("Polymer "+name,rgb,.08,.31,True)
        box("Block "+name,(0,0,0),(.032,.032,.032),m,root,bevel=.0011)
        lettering("P%02d"%(i+1),(-.011,-.011,.01604),.0034,ink,root)
    tool=empty("Cartesian tool")
    bodies["tool"]=tool
    for z,r,d in [(-.008,.011,.002),(-.006,.008,.003),(-.004,.010,.002),(-.002,.007,.002)]:
        cylinder("Vacuum bellows",(0,0,z),r,d,black,tool)
    cylinder("Cup collar",(0,0,.002),.007,.006,brass,tool,6)
    cylinder("Vacuum stem",(0,0,.014),.005,.019,steel,tool)
    cylinder("Stem nut",(0,0,.021),.009,.005,steel,tool,6)
    box("Tool carriage",(0,0,.035),(.026,.024,.020),alloy,tool,bevel=.0017)
    for x in (-.009,.009):
        for y in (-.008,.008): screw(x,y,.045,tool,.75)
    cylinder("Push-in fitting",(.005,-.002,.053),.0048,.011,brass,tool,6)
    cylinder("Collet",(.005,-.002,.059),.0044,.003,black,tool)
    hose([(.005,-.002,.06),(.006,-.002,.08),(.014,-.018,.09),(.021,-.063,.086),
          (.018,-.109,.093)],.0025,blue,tool)
    box("Wrist bracket",(-.024,-.024,.055),(.004,.05,.007),alloy,tool)
    # Lens aperture faces the recorded optical axis. The housing sits behind the entrance pupil.
    bracket=empty("Camera mount")
    bracket.parent, bracket.location = tool,(0,-.052,.065)
    bracket.rotation_euler.x=math.radians(30)
    box("Camera housing",(0,0,.014),(.029,.025,.023),black,bracket,bevel=.0016)
    cylinder("Lens barrel",(0,0,.002),.0085,.004,alloy,bracket)
    cylinder("Lens glass",(0,0,-.0001),.0065,.0004,glass,bracket)
    # Visual housing for the Cartesian stage. The wrist camera and TCP remain on the measured body.
    stage=empty("Y-axis carriage")
    for x in (-.275,.275):
        box("Bridge support",(x,-.223,.086),(.031,.035,.172),alloy,bevel=.002)
    box("X rail",(0,-.223,.178),(.58,.029,.023),alloy,bevel=.0016)
    for x in (-.014,.014):
        box("Slide channel",(x,-.188,0),(.005,.40,.012),steel,stage)
    box("Slide beam",(0,-.188,.004),(.026,.40,.013),alloy,stage)
    box("Slide attachment",(0,0,.002),(.038,.030,.020),alloy,stage)
    box("Tool side mount",(-.020,0,.039),(.042,.013,.011),alloy,tool)
    cylinder("Vertical quill",(-.04,0,.107),.0068,.139,steel,tool)
    return bodies,stage


def lighting(scene):
    world=bpy.data.worlds.new("Workshop daylight")
    scene.world=world
    world.use_nodes=True
    env=world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    env.image=bpy.data.images.load(str(ASSETS/"workshop_1k.hdr"))
    bg=world.node_tree.nodes.get("Background")
    world.node_tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value=.65
    bpy.ops.object.light_add(type="AREA",location=(-.22,-.25,.60))
    soft=bpy.context.object
    soft.rotation_euler=(-soft.location).to_track_quat("-Z","Y").to_euler()
    soft.data.energy, soft.data.shape, soft.data.size, soft.data.size_y=9,"RECTANGLE",.50,.27


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode",type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--width",type=int,default=640)
    p.add_argument("--samples",type=int,default=32)
    p.add_argument("--threads",type=int,default=12)
    p.add_argument("--device",choices=("CPU","OPTIX","CUDA"),default="CPU")
    p.add_argument("--view",choices=("overhead","wrist"),required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--still",type=int,help="render one frame to PNG")
    mode.add_argument("--keyframes",action="store_true",help="write one JPEG per distinct tool-call boundary")
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
    a=p.parse_args(argv)
    if a.width%8 or min(a.width,a.samples,a.threads)<=0: p.error("positive counts; width must be divisible by eight")
    a.out.mkdir(parents=True,exist_ok=True)
    task=json.loads((a.episode/"task.json").read_text())
    records=[json.loads(x) for x in (a.episode/"poses.jsonl").read_text().splitlines()]
    cal=json.loads((a.episode/"cameras.json").read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    global black,steel,alloy,brass,ink,blue,glass
    black=material("Nitrile",(.024,.029,.034),.1,.44,True)
    steel=material("Brushed steel",(.45,.46,.46),.93,.27,True)
    alloy=material("Anodized aluminum",(.23,.265,.29),.8,.29,True)
    brass=material("Brass",(.61,.38,.13),.85,.31)
    ink=material("Pale marking",(.8,.8,.72),.05,.58)
    blue=material("Blue air hose",(.018,.19,.42),.08,.27)
    glass=material("Optical coating",(.015,.024,.037),.4,.09)
    bodies,stage=build(task)
    s=bpy.context.scene
    lighting(s)
    cam_data=bpy.data.cameras.new(a.view)
    cam=bpy.data.objects.new(a.view,cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.type,cam_data.sensor_fit="PERSP","HORIZONTAL"
    cam_data.lens=cam_data.sensor_width*cal["views"][a.view]["K"][0][0]/cal["resolution"][0]
    cam_data.clip_start=.001
    s.camera=cam
    s.render.engine="CYCLES"
    if a.device!="CPU":
        prefs=bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type=a.device
        prefs.refresh_devices()
        available=[d for d in prefs.devices if d.type==a.device]
        if not available: raise RuntimeError(f"No visible {a.device} device")
        for d in prefs.devices: d.use=d in available
        s.cycles.device="GPU"
    s.cycles.samples=a.samples
    s.cycles.use_denoising=True
    s.cycles.max_bounces=5
    s.render.threads_mode="FIXED"
    s.render.threads=a.threads
    s.render.resolution_x,s.render.resolution_y,s.render.resolution_percentage=a.width,a.width*3//4,100
    s.render.use_persistent_data=True
    s.render.image_settings.file_format="PNG"
    frames=[records[a.still]] if a.still is not None else records
    if a.keyframes:
        calls=[json.loads(line) for line in (a.episode/"tool_calls.jsonl").read_text().splitlines()]
        selected={0} | {c["after_frame"] for c in calls}
        frames=[row for row in records if row["frame"] in selected]
        dest_dir=a.out / a.view
        dest_dir.mkdir(exist_ok=False)
        s.render.image_settings.file_format="JPEG"
        s.render.image_settings.quality=95
    dest=a.out/(a.view+".mp4" if a.still is None else f"{a.view}_{a.still:05d}.png")
    if not a.keyframes and dest.exists(): raise FileExistsError(dest)
    import tempfile
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        for row in frames:
            for name,pose in row["body_xyz_quat_wxyz"].items():
                b=bodies[name]
                b.location,b.rotation_mode,b.rotation_quaternion=pose[:3],"QUATERNION",pose[3:]
            toolpos=row["body_xyz_quat_wxyz"]["tool"]
            stage.location=(toolpos[0]-.04,toolpos[1],.18)
            cam.matrix_world=Matrix(row["camera_to_world"][a.view])
            s.render.filepath=str(dest.resolve() if a.still is not None else Path(tmp).resolve()/f"{row['frame']:05d}.png")
            if a.keyframes:
                s.render.filepath=str(dest_dir.resolve()/f"{row['frame']:05d}.jpg")
            bpy.ops.render.render(write_still=True)
        if a.still is None and not a.keyframes:
            movie=Path(tmp)/"render.mp4"
            subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-framerate",str(cal["fps"]),"-i",
                            str(Path(tmp)/"%05d.png"),"-c:v","libx264","-crf","16","-pix_fmt","yuv420p",
                            "-threads","2","-movflags","+faststart",str(movie)],check=True)
            movie.replace(dest)


if __name__=="__main__":
    main()
