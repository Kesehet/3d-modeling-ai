from app.main import HardSurfaceCageSpec, _normalize_hard_surface_cage_payload


def test_hard_surface_cage_normalizes_human_style_half_profiles():
    payload = _normalize_hard_surface_cage_payload(
        {
            "title": "Reference-built hard surface",
            "length_axis": "y",
            "sections": [
                {
                    "position": -3.0,
                    "profile": [
                        {"half_width": 0.4, "height": -0.5},
                        {"half_width": 1.1, "height": -0.3},
                        {"half_width": 1.0, "height": 0.5},
                        {"half_width": 0.6, "height": 1.0},
                        {"half_width": 0.3, "height": 1.2},
                    ],
                },
                {
                    "position": -1.0,
                    "profile": [[0.3, -0.5], [1.2, -0.3], [1.1, 0.6], [0.7, 1.2], [0.2, 1.35]],
                },
                {
                    "position": 1.0,
                    "profile": [[0.3, -0.5], [1.2, -0.3], [1.1, 0.6], [0.7, 1.2], [0.2, 1.35]],
                },
                {
                    "position": 3.0,
                    "profile": [[0.4, -0.5], [1.1, -0.3], [1.0, 0.5], [0.6, 1.0], [0.3, 1.2]],
                },
            ],
            "boolean_cutters": [
                {
                    "name": "opening",
                    "type": "cylinder",
                    "center": [1.0, -1.5, -0.2],
                    "size": [0.7, 0.4, 0.7],
                    "rotation": [0, 90, 0],
                }
            ],
        },
        "fallback",
    )

    spec = HardSurfaceCageSpec.model_validate(payload)

    assert spec.axis == "y"
    assert len(spec.stations) == 4
    assert all(len(station.profile) == 5 for station in spec.stations)
    assert all(station.profile[0][0] == 0.0 for station in spec.stations)
    assert all(station.profile[-1][0] == 0.0 for station in spec.stations)
    assert spec.cutters[0].shape == "cylinder"


def test_hard_surface_cage_resamples_inconsistent_profile_rows():
    payload = _normalize_hard_surface_cage_payload(
        {
            "stations": [
                {"position": -3, "profile": [[0, -1], [1, -0.5], [1, 0.5], [0, 1]]},
                {"position": -1, "profile": [[0, -1], [1, -0.5], [1, 0.5], [0, 1]]},
                {"position": 0, "profile": [[0, -1], [1, -0.7], [1.2, 0], [1, 0.7], [0, 1]]},
                {"position": 1, "profile": [[0, -1], [1, -0.5], [1, 0.5], [0, 1]]},
                {"position": 3, "profile": [[0, -1], [1, -0.5], [1, 0.5], [0, 1]]},
            ]
        },
        "object",
    )

    spec = HardSurfaceCageSpec.model_validate(payload)

    assert len(spec.stations) == 5
    assert all(len(station.profile) == 4 for station in spec.stations)


def test_hard_surface_cage_expands_three_terse_cross_sections():
    payload = _normalize_hard_surface_cage_payload(
        {
            "output": {
                "length_axis": "y",
                "cross_sections": [
                    {
                        "axis_position": -3,
                        "half_profile": [[0, -0.5], [1.0, -0.3], [0.8, 0.7], [0, 1.1]],
                    },
                    {
                        "axis_position": 0,
                        "half_profile": [[0, -0.5], [1.2, -0.3], [0.9, 0.9], [0, 1.4]],
                    },
                    {
                        "axis_position": 3,
                        "half_profile": [[0, -0.5], [1.0, -0.3], [0.8, 0.7], [0, 1.1]],
                    },
                ],
            }
        },
        "object",
    )

    spec = HardSurfaceCageSpec.model_validate(payload)

    assert len(spec.stations) == 5
    assert [station.position for station in spec.stations] == [-3.0, -1.5, 0.0, 1.5, 3.0]
