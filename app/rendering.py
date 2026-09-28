"""Shared Blender camera framing for useful, consistently labeled QA images."""


def camera_framing_script() -> str:
    return r'''
    camera.data.type = "ORTHO"
    # Center the orthographic camera on the authored geometry. Fit its projected
    # bounds for each direction rather than rendering a tiny object at a fixed
    # perspective distance. No subject-specific framing rules are used.
    if "-" not in name and name != "top":
        camera.location.z = center.z
    elif "-" in name:
        camera.location.z = center.z + distance * 0.48
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    inverse = camera.rotation_euler.to_matrix().transposed()
    projected = [inverse @ (point - center) for point in points]
    projected_width = max(p.x for p in projected) - min(p.x for p in projected)
    projected_height = max(p.y for p in projected) - min(p.y for p in projected)
    camera.data.ortho_scale = max(projected_width, projected_height, 0.01) * 1.18
    camera.data.clip_end = max(1000.0, distance * 4)
'''
