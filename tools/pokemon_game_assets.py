import bpy, math, sys
from pathlib import Path

OUT = Path(sys.argv[sys.argv.index("--")+1] if "--" in sys.argv else "generated/assets").resolve()
OUT.mkdir(parents=True, exist_ok=True)

def clear():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.materials): bpy.data.materials.remove(m)

def mat(name,c,emit=0):
    m=bpy.data.materials.new(name); m.diffuse_color=(*c,1); m.use_nodes=True
    b=m.node_tree.nodes.get("Principled BSDF"); b.inputs["Base Color"].default_value=(*c,1); b.inputs["Roughness"].default_value=.78
    if emit and "Emission" in b.inputs: b.inputs["Emission"].default_value=(*c,1)
    if emit and "Emission Strength" in b.inputs: b.inputs["Emission Strength"].default_value=emit
    return m

def sph(n,loc,sc,ma):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, location=loc); o=bpy.context.object; o.name=n; o.scale=sc
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); o.data.materials.append(ma)
    for p in o.data.polygons:p.use_smooth=False
    return o

def cube(n,loc,sc,ma,rot=(0,0,0)):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc,rotation=rot);o=bpy.context.object;o.name=n;o.scale=sc
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);o.data.materials.append(ma);return o

def cone(n,loc,r,d,ma,rot=(0,0,0),v=7):
    bpy.ops.mesh.primitive_cone_add(vertices=v,radius1=r,radius2=0,depth=d,location=loc,rotation=rot);o=bpy.context.object;o.name=n;o.data.materials.append(ma);return o

def cyl(n,loc,r,d,ma,rot=(0,0,0),v=10):
    bpy.ops.mesh.primitive_cylinder_add(vertices=v,radius=r,depth=d,location=loc,rotation=rot);o=bpy.context.object;o.name=n;o.data.materials.append(ma);return o

def eyes(y,z,s=.07):
    black=mat("black",(0.02,0.02,0.025))
    sph("eyeL",(-.18,y,z),(s,s*.65,s*1.15),black); sph("eyeR",(.18,y,z),(s,s*.65,s*1.15),black)

def export(name):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(OUT/f"{name}.glb"),export_format="GLB",export_apply=True)

def gecko(name,stage):
    clear(); g=mat("green",(0.18,0.68-.06*stage,0.28)); l=mat("light",(0.40,0.90,0.48)); r=mat("red",(0.82,0.10,0.13)); s=1+stage*.2
    sph("body",(0,0,.95*s),(.48*s,.38*s,.72*s),g); sph("head",(0,-.03,1.72*s),(.54*s,.48*s,.48*s),l); eyes(-.44*s,1.8*s,.065*s)
    sph("footL",(-.28*s,-.05,.22*s),(.25*s,.35*s,.13*s),r); sph("footR",(.28*s,-.05,.22*s),(.25*s,.35*s,.13*s),r)
    cube("tail",(0,.52*s,.9*s),(.45*s,.13*s,.8*s),g,(math.radians(-28),0,0))
    if stage>0:
        cone("headleaf",(0,.05,2.25*s),.24*s,.85*s,g,(0,0,.15)); cone("armL",(-.62*s,0,1.03*s),.18*s,.65*s,g,(0,math.pi/2,0)); cone("armR",(.62*s,0,1.03*s),.18*s,.65*s,g,(0,-math.pi/2,0))
    if stage>1:
        y=mat("yellow",(0.93,.82,.22)); sph("chest",(0,-.39*s,1.08*s),(.18*s,.10*s,.28*s),y)
    export(name)

def croc(name,stage):
    clear(); b=mat("blue",(0.18,.62+.05*stage,.88)); cream=mat("cream",(.96,.85,.58)); red=mat("crest",(.88,.08,.1)); s=1+stage*.2
    sph("body",(0,0,.9*s),(.55*s,.45*s,.75*s),b); sph("head",(0,-.03,1.7*s),(.60*s,.55*s,.48*s),b); eyes(-.5*s,1.78*s,.065*s)
    cube("jaw",(0,-.48*s,1.52*s),(.38*s,.18*s,.13*s),cream)
    for i in range(3+stage): cone(f"crest{i}",(0,.30*s,(1.85+i*.16)*s),.13*s,.45*s,red,(math.pi/2,0,0),5)
    export(name)

def dragon(name,stage):
    clear(); o=mat("orange",(.94-.08*stage,.36+.05*stage,.12)); c=mat("cream",(.98,.75,.36)); flame=mat("flame",(1,.65,.04),2); s=1+stage*.22
    sph("body",(0,0,.9*s),(.52*s,.42*s,.76*s),o); sph("head",(0,-.04,1.72*s),(.55*s,.48*s,.48*s),o); eyes(-.45*s,1.8*s,.065*s)
    sph("belly",(0,-.39*s,.9*s),(.31*s,.10*s,.52*s),c); cyl("tail",(.52,.25,.78*s),.13*s,1.3*s,o,(math.pi/2,0,-.8)); sph("flame",(1.03,.15,.92*s),(.18*s,.18*s,.3*s),flame)
    if stage==1: cone("horn",(0,.05,2.18*s),.18*s,.55*s,o)
    if stage>1:
        wing=mat("wing",(.12,.45,.62))
        cube("wingL",(-.75,.18,1.25*s),(.55,.08,.65),wing,(0,.1,-.45)); cube("wingR",(.75,.18,1.25*s),(.55,.08,.65),wing,(0,-.1,.45))
    export(name)

def lucario(name,mega=False):
    clear(); blue=mat("blue",(.10,.34 if mega else .45,.80)); black=mat("black",(.03,.04,.05)); cream=mat("cream",(.92,.72,.34)); red=mat("red",(.78,.06,.08)); s=1.16 if mega else 1
    sph("body",(0,0,.95*s),(.46*s,.36*s,.75*s),blue); sph("head",(0,-.03,1.78*s),(.48*s,.44*s,.45*s),blue); eyes(-.42*s,1.84*s,.055*s)
    cone("earL",(-.26,0,2.26*s),.17*s,.75*s,black,(0,0,-.12)); cone("earR",(.26,0,2.26*s),.17*s,.75*s,black,(0,0,.12))
    sph("chest",(0,-.36*s,.98*s),(.22*s,.10*s,.42*s),cream); cone("chestSpike",(0,-.66*s,1.05*s),.11*s,.46*s,cream,(math.pi/2,0,0),8)
    if mega:
        cone("shoulderL",(-.55,-.02,1.22*s),.14*s,.55*s,red,(0,math.pi/2,0)); cone("shoulderR",(.55,-.02,1.22*s),.14*s,.55*s,red,(0,-math.pi/2,0))
        cone("tail",(0,.55,.85*s),.16*s,1.25*s,blue,(math.pi/2,0,0))
    export(name)

def trainer():
    clear(); skin=mat("skin",(.88,.62,.43)); blue=mat("jacket",(.10,.28,.72)); red=mat("cap",(.78,.05,.06)); dark=mat("pants",(.05,.07,.10))
    cyl("body",(0,0,1.0),.36,1.1,blue); sph("head",(0,0,1.8),(.37,.34,.38),skin); cube("cap",(0,-.02,2.11),(.42,.36,.09),red); cyl("legL",(-.16,0,.38),.11,.7,dark); cyl("legR",(.16,0,.38),.11,.7,dark)
    export("trainer")

def gym():
    clear(); wall=mat("wall",(.86,.90,.94)); roof=mat("roof",(.72,.08,.09)); door=mat("door",(.08,.22,.48)); gold=mat("gold",(1,.72,.08))
    cube("building",(0,0,2.2),(4.2,2.6,2.2),wall); cube("roof",(0,0,4.55),(4.65,2.95,.35),roof); cube("door",(0,-2.63,1.4),(.8,.12,1.4),door)
    cube("sign",(0,-2.86,3.7),(1.15,.10,.45),gold)
    export("gym")

gecko("treecko",0); gecko("grovyle",1); gecko("sceptile",2)
croc("totodile",0); croc("croconaw",1); croc("feraligatr",2)
dragon("charmander",0); dragon("charmeleon",1); dragon("charizard",2)
lucario("lucario",False); lucario("mega_lucario",True); trainer(); gym()
print("DONE", OUT)
