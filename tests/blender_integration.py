"""Real Blender/export regression, executed by the production Docker CI job."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

from app.cage_edits import CageEditAction, apply_cage_edit_action
from app.component_assembly import component_assembly_script
from app.generic_builder import generic_scene_script
from app.hard_surface_builder import hard_surface_cage_script
from app.mesh_builder import adaptive_loft_script


def run(root, name, code, extra):
    target = root / name
    target.mkdir()
    args = {"spec": {}, "prefix": "model-v1", "blend_path": str(target / "model-v1.blend"),
            "output_dir": str(target), "exports_dir": str(target),
            "qa_path": str(target / "model-v1-qa.json"), **extra}
    script = target / "build.py"
    script.write_text("args = " + repr(args) + "\n" + code)
    result = subprocess.run([os.environ["BLENDER_BIN"], "--background", "--factory-startup",
                             "--threads", "2", "--python-exit-code", "1", "--python", str(script)],
                            text=True, capture_output=True, timeout=240, check=False)
    (target / "blender.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (target / "model-v1.blend").stat().st_size > 1000
    assert (target / "model-v1.glb").read_bytes()[:4] == b"glTF"
    assert (target / "model-v1.obj").stat().st_size > 100
    assert (target / "model-v1.stl").stat().st_size > 100
    assert len(list(target.glob("model-v1-*.png"))) == 9
    for path in target.glob("model-v1-*.png"):
        with Image.open(path) as image:
            image = image.convert("RGB")
            assert image.size == (640, 640)
            background = Image.new("RGB", image.size, image.getpixel((0, 0)))
            silhouette = ImageChops.difference(image, background).convert("L").point(
                lambda value: 255 if value > 15 else 0,
            ).getbbox()
            assert silhouette is not None, (name, path.name, "empty render")
            x0, y0, x1, y1 = silhouette
            assert max(x1 - x0, y1 - y0) > 320, (name, path.name, silhouette)
    qa = json.loads((target / "model-v1-qa.json").read_text())
    assert qa["mesh_object_count"] > 0
    assert qa["non_manifold_edges"] == 0, qa
    if "applied_cutters" in extra:
        assert qa["applied_cutters"] == extra["applied_cutters"], qa
    print(name, "passed:", qa)
    return target


def main():
    root = Path(os.environ.get("BLENDER_TEST_OUTPUT") or tempfile.mkdtemp(prefix="blender-regression-"))
    root.mkdir(parents=True, exist_ok=True)
    spec = {"title": "Generic editable body", "axis": "y", "subdivision_levels": 0,
            "bevel_width": 0.01, "presentation_base": False,
            "stations": [{"position": p, "profile": [[0, 0], [1, 0], [1, 1], [0, 1]]}
                         for p in (-2, -1, 1, 2)], "cutters": [], "attachments": []}
    parent = run(root, "cage", hard_surface_cage_script(), {"spec": spec})
    # Exact declarative geometry captured from the live job that exported 64
    # non-manifold edges with no cutters. This fixture is never used by generation.
    live_spec = json.loads((Path(__file__).parent / "fixtures/cage_live_regression.json").read_text())
    run(root, "live_shell", hard_surface_cage_script(), {"spec": live_spec})
    before = (parent / "model-v1.blend").read_bytes()
    edited = apply_cage_edit_action(spec, CageEditAction(
        operation="add_cutter", reason="Open a transverse hole",
        part={"name": "opening", "shape": "cylinder", "location": [0, 0, 0.5],
              "scale": [0.25, 0.25, 1.5], "rotation_deg": [0, 90, 0]},
    ))
    run(root, "edited", hard_surface_cage_script(), {"spec": edited, "applied_cutters": 1})
    assert (parent / "model-v1.blend").read_bytes() == before
    # A tapered shell with subdivision exercises both mirror axes and the cap
    # normals that previously produced holes under bevel on live body models.
    tapered = {**spec, "axis": "x", "subdivision_levels": 1, "stations": [
        {"position": p, "profile": [[0, 0], [w, 0.1], [w, h * 0.7], [w * 0.6, h], [0, h]]}
        for p, w, h in [(-3, 0.6, 0.4), (-2, 0.9, 0.7), (-1, 1, 1.3), (1, 1, 1.4), (2, 0.8, 1), (3, 0.6, 0.6)]
    ]}
    run(root, "tapered", hard_surface_cage_script(), {"spec": tapered})
    run(root, "loft", adaptive_loft_script(), {"spec": {
        "title": "Loft", "axis": "z", "presentation_base": False,
        "sections": [{"position": p, "contour": [[-w, -w], [w, -w], [w, w], [-w, w]]}
                     for p, w in [(0, 1), (1, 0.8), (2, 0.3)]],
    }})
    child = run(root, "primitive", generic_scene_script(), {"spec": {
        "title": "Generic attachment", "presentation_base": False,
        "objects": [{"name": "part", "shape": "cube", "location": [0, 0, 0],
                     "scale": [0.3, 0.3, 0.3], "color": "#567890"}],
    }})
    for changed in (False, True):
        run(root, "scoped_proof_changed" if changed else "scoped_proof_unchanged", generic_scene_script() + r"""
proof = qa["protected_part_proof"]
assert proof["verified"] is args["expected_preserved"], proof
""", {
            "spec": {"title": "Local repair proof", "presentation_base": False, "objects": [
                {"name": "part", "shape": "cube", "location": [0,0,0],
                 "scale": [.5,.3,.3] if changed else [.3,.3,.3], "color": "#567890"},
                {"name": "new_contact", "shape": "cube", "location": [1,0,0], "scale": [.1,.1,.1]},
            ]},
            "preservation_context": {"baseline_blend_path": str(child / "model-v1.blend"),
                                     "protected_object_names": ["part"]},
            "expected_preserved": not changed,
        })
    run(root, "assembly", component_assembly_script(), {
        "parent_blend_path": str(parent / "model-v1.blend"),
        "component_blend_path": str(child / "model-v1.blend"), "component_name": "part",
        "instances": [{"location": [0, 0, 1], "scale": [0.5, 0.5, 0.5],
                       "rotation_deg": [0, 0, 0]}],
    })
    anchored_child = run(root, "anchored_component", generic_scene_script(), {"spec": {
        "title": "Origin-anchored generic component", "presentation_base": False,
        "objects": [{"name": "offset_part", "shape": "cube", "location": [1, 0, 0],
                     "scale": [0.2, 0.2, 0.2], "color": "#456789", "bevel": False}],
    }})
    run(root, "origin_anchor_assembly", component_assembly_script() + r"""
assert len(installed_objects) == 1, len(installed_objects)
installed = installed_objects[0]
world_points = [installed.matrix_world @ Vector(corner) for corner in installed.bound_box]
geometry_center_x = (min(point.x for point in world_points) + max(point.x for point in world_points)) / 2
# The child geometry is centered at child-space x=1. Installing the child global
# origin at parent x=2 must therefore put the geometry at x=3. Re-centering or
# losing the unlinked source transform would incorrectly put it at x=2.
assert abs(geometry_center_x - 3.0) < 0.001, geometry_center_x
""", {
        "parent_blend_path": str(parent / "model-v1.blend"),
        "component_blend_path": str(anchored_child / "model-v1.blend"),
        "component_name": "origin_test",
        "instances": [{"location": [2, 0, 0], "scale": [1, 1, 1],
                       "rotation_deg": [0, 0, 0]}],
    })
    panel = {"name": "Shaped panel", "shape": "mesh", "location": [0, 0, 1], "scale": [1, 1, 1],
             "color": "#123456", "bevel": False,
             "vertices": [[-.3,-.2,0],[.3,-.2,0],[.2,.2,0],[-.2,.2,0],
                          [-.3,-.2,.1],[.3,-.2,.1],[.2,.2,.1],[-.2,.2,.1]],
             "faces": [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]]}
    profile_child = run(root, "section_profile_child", generic_scene_script() + r"""
modifier = objects[0].modifiers.new("Measured array", "ARRAY")
modifier.count = 2
modifier.use_relative_offset = False
modifier.use_constant_offset = True
modifier.constant_offset_displace = (.4, 0, 0)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=BLEND)
""", {"spec": {
        "title": "Tapered offset panel", "presentation_base": False,
        "objects": [{**panel, "location": [0, 1, 0], "vertices": [
            [-.8,0,-.05],[.8,0,-.05],[.3,4,-.05],[-.3,4,-.05],
            [-.8,0,.05],[.8,0,.05],[.3,4,.05],[-.3,4,.05],
        ]}],
    }})
    run(root, "section_profile_assembly", component_assembly_script() + r"""
profile = qa["component_width_profiles"][0]
assert profile["long_axis"] == "Y", profile
root_widths = [profile["root_width"]]
distal_widths = [profile["distal_width"]]
assert abs(root_widths[0] - 2.4) < .001, root_widths
assert abs(distal_widths[0] - 3.6) < .001, distal_widths
assert profile["direction"] == "widens_outward", profile
assert all(p["measured"] and p["direction"] == "widens_outward" for p in qa["component_width_profiles"])
assert qa["parent_geometry_preserved"] is True
""", {
        "parent_blend_path": str(parent / "model-v1.blend"),
        "component_blend_path": str(profile_child / "model-v1.blend"),
        "component_name": "measured_panel", "long_axis_sign": "negative", "repeated_axis_alignment": "radial",
        "instances": [
            {"location": [5,0,0], "scale": [2,1,3], "rotation_deg": [0,0,90]},
            {"location": [-2.5,4.330127,0], "scale": [2,1,3], "rotation_deg": [0,0,210]},
            {"location": [-2.5,-4.330127,0], "scale": [2,1,3], "rotation_deg": [0,0,330]},
        ],
    })
    run(root, "mesh_part", generic_scene_script(), {"spec": {
        "title": "Polygon panel", "presentation_base": False, "objects": [panel]}})
    run(root, "cage_panel", hard_surface_cage_script(), {"spec": {**spec, "attachments": [panel]}})
    run(root, "hollow_revolve", generic_scene_script() + r"""
body = objects[0]
hit, location, normal, index = body.ray_cast(Vector((0, 0, 3)), Vector((0, 0, -1)))
assert hit and abs(location.z - 0.2) < 0.001, (hit, location)
""", {"spec": {"title": "Closed revolved material profile", "presentation_base": False,
                "objects": [{"name": "profile", "shape": "lathe", "location": [0, 0, 0],
                             "profile": [[0,0],[1,0],[1,2],[0.8,2],[0.8,0.2],[0,0.2]],
                             "bevel": False}]}})
    run(root, "path_sweep", generic_scene_script(), {"spec": {
        "title": "Curved tube", "presentation_base": False,
        "objects": [{"name": "path", "shape": "sweep", "location": [0,0,0],
                     "path": [[-1,0,-1],[-1.8,0,0],[-1,0,1]], "radius": 0.1, "bevel": False}],
    }})
    run(root, "full_dimensions", generic_scene_script() + r"""
body = objects[0]
points = [body.matrix_world @ Vector(corner) for corner in body.bound_box]
actual = [max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)]
assert all(abs(a-b) < 0.001 for a,b in zip(actual, [0.3,2.5,1.5])), actual
""", {"spec": {"title": "Sized and rotated block", "presentation_base": False,
                "objects": [{"name": "block", "shape": "cube", "location": [0,0,0],
                             "dimensions": [0.3,1.5,2.5], "rotation_deg": [90,0,0], "bevel": False}]}})
    run(root, "primitive_radius", generic_scene_script() + r"""
body = objects[0]
points = [body.matrix_world @ Vector(corner) for corner in body.bound_box]
actual = [max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)]
assert abs(actual[0] - 0.16) < 0.002, actual
assert abs(actual[1] - 0.16) < 0.002, actual
assert abs(actual[2] - 2.0) < 0.002, actual
""", {"spec": {
        "title": "Explicit-radius cylinder",
        "presentation_base": False,
        "objects": [{
            "name": "shaft",
            "shape": "cylinder",
            "location": [0,0,0],
            "radius": 0.08,
            "scale": [1,1,1],
            "bevel": False,
        }],
    }})
    run(root, "sphere_explicit_radius", generic_scene_script() + r"""
body = objects[0]
points = [body.matrix_world @ Vector(corner) for corner in body.bound_box]
actual = [max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)]
assert all(abs(value - 0.6) < 0.004 for value in actual), actual
""", {"spec": {
        "title": "Explicitly sized round primitive",
        "presentation_base": False,
        "objects": [{"name": "round_part", "shape": "sphere", "location": [0, 0, 0],
                     "radius": 0.3, "bevel": False}],
    }})
    run(root, "procedural_boolean_cutout", generic_scene_script() + r"""
body = objects[0]
# A subtractive rim must split shading normals; smooth shading may not warp
# the flat outer surface toward the cavity wall. This tests evaluated normals,
# not merely the presence of a modifier or a particular implementation.
assert any(edge.use_edge_sharp for edge in body.data.edges)
body.data.update()
top_faces = [face for face in body.data.polygons if face.normal.z > 0.999]
assert top_faces
for face in top_faces:
    for loop_index in face.loop_indices:
        normal = body.data.corner_normals[loop_index].vector
        assert normal.z > 0.999, (face.index, loop_index, tuple(normal))
center_hit, center_location, center_normal, center_index = body.ray_cast(
    Vector((0, 0, 2)), Vector((0, 0, -1))
)
edge_hit, edge_location, edge_normal, edge_index = body.ray_cast(
    Vector((0.75, 0, 2)), Vector((0, 0, -1))
)
assert not center_hit, (center_hit, center_location)
assert edge_hit and abs(edge_location.z - 0.5) < 0.08, (edge_hit, edge_location)
""", {"spec": {
        "title": "Boolean socket",
        "presentation_base": False,
        "objects": [{
            "name": "body", "shape": "cube", "location": [0,0,0],
            "dimensions": [2,2,1], "bevel": False,
        }],
        "cutters": [{
            "name": "socket", "target": "body", "shape": "cylinder",
            "location": [0,0,0], "dimensions": [0.5,0.5,2],
        }],
    }, "applied_cutters": 1})
    run(root, "procedural_radial_cutters", generic_scene_script() + r"""
body = objects[0]
for angle_deg in (0, 120, 240):
    angle = math.radians(angle_deg)
    x = 1.2 * math.cos(angle)
    y = 1.2 * math.sin(angle)
    hit, location, normal, index = body.ray_cast(
        Vector((x, y, 2)), Vector((0, 0, -1))
    )
    assert not hit, (angle_deg, hit, location)
center_hit, center_location, center_normal, center_index = body.ray_cast(
    Vector((0, 0, 2)), Vector((0, 0, -1))
)
assert center_hit and abs(center_location.z - 0.4) < 0.08, (center_hit, center_location)
""", {"spec": {
        "title": "Radial socket pattern",
        "presentation_base": False,
        "objects": [{
            "name": "body", "shape": "cylinder", "location": [0,0,0],
            "dimensions": [4,4,0.8], "bevel": False,
        }],
        "cutters": [{
            "name": "socket", "target": "body", "shape": "cylinder",
            "location": [1.2,0,0], "dimensions": [0.35,0.35,2],
            "radial_repeat_count": 3,
            "radial_repeat_axis": "z",
            "radial_repeat_center": [0,0,0],
        }],
    }, "applied_cutters": 3})
    print("BLENDER_INTEGRATION_OK", root)


if __name__ == "__main__":
    main()
