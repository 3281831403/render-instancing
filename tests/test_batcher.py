"""render-instancing 的 15 个红灯测试。

每个测试至少调用一个 ``instancing.batcher`` 的公开桩，并表达一个用户可见
契约。实例一律使用内联四字段字典 ``{"mesh", "material", "matrix", "visible"}``；
矩阵是 4×4 行主序扁平列表，平移在下标 3、7、11。
"""

import pytest

from instancing.batcher import (
    EPS,
    MAX_INDEX,
    InstanceError,
    cull_instances,
    dedupe_instances,
    group_instances,
    instance_batch,
    instance_index,
    pack_matrices,
    sort_by_material,
    validate_instances,
)


def make_matrix(tx=0.0, ty=0.0, tz=0.0, scale=1.0):
    """Return a row-major 4×4 identity matrix with translation and scale."""
    return [
        scale,
        0.0,
        0.0,
        tx,
        0.0,
        scale,
        0.0,
        ty,
        0.0,
        0.0,
        scale,
        tz,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def make_instance(mesh="grass", material="mat-a", matrix=None, visible=True):
    """Build the documented four-field instance dictionary."""
    return {
        "mesh": mesh,
        "material": material,
        "matrix": make_matrix() if matrix is None else matrix,
        "visible": visible,
    }


def test_group_instances_preserves_first_seen_order():
    grass_a = make_instance("grass", "mat-a", make_matrix(tx=0.0))
    rock_b = make_instance("rock", "mat-b", make_matrix(tx=1.0))
    grass_b = make_instance("grass", "mat-b", make_matrix(tx=2.0))
    grass_a_again = make_instance("grass", "mat-a", make_matrix(tx=3.0))
    grass_b_again = make_instance("grass", "mat-b", make_matrix(tx=4.0))

    groups = group_instances(
        [grass_a, rock_b, grass_b, grass_a_again, grass_b_again]
    )

    assert list(groups) == [
        ("grass", "mat-a"),
        ("rock", "mat-b"),
        ("grass", "mat-b"),
    ]
    assert len(groups[("grass", "mat-a")]) == 2
    assert groups[("grass", "mat-a")][0] is grass_a
    assert groups[("grass", "mat-a")][1] is grass_a_again
    assert groups[("rock", "mat-b")] == [rock_b]
    assert groups[("grass", "mat-b")][0] is grass_b
    assert groups[("grass", "mat-b")][1] is grass_b_again

    assert list(
        group_instances(
            [grass_a, rock_b, grass_b, grass_a_again, grass_b_again]
        )
    ) == list(groups)


def test_pack_limits_by_instance_count_not_bytes():
    group = [
        make_instance(material="mat-a", matrix=make_matrix(tx=float(index)))
        for index in range(5)
    ]

    result = pack_matrices(group, max_instances=2)
    batches = result["batches"]

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert all(len(matrix) == 16 for batch in batches for matrix in batch)
    assert [[matrix[3] for matrix in batch] for batch in batches] == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0],
    ]


def test_cull_keeps_bounding_sphere_crossing_plane():
    crossing = make_instance(matrix=make_matrix(tx=-0.5))
    outside = make_instance(matrix=make_matrix(tx=-2.0))
    inside = make_instance(matrix=make_matrix(tx=2.0))
    plane = (1.0, 0.0, 0.0, 0.0)

    kept = cull_instances([crossing, outside, inside], [plane])

    assert len(kept) == 2
    assert kept[0] is crossing
    assert kept[1] is inside


def test_sort_by_material_is_stable():
    stone_a = make_instance("stone", "stone", make_matrix(tx=0.0))
    grass_a = make_instance("grass", "grass", make_matrix(tx=1.0))
    stone_b = make_instance("stone", "stone", make_matrix(tx=2.0))
    metal = make_instance("metal", "metal", make_matrix(tx=3.0))
    grass_b = make_instance("grass", "grass", make_matrix(tx=4.0))
    stone_c = make_instance("stone", "stone", make_matrix(tx=5.0))

    ordered = sort_by_material(
        [stone_a, grass_a, stone_b, metal, grass_b, stone_c]
    )

    expected = [grass_a, grass_b, metal, stone_a, stone_b, stone_c]
    assert all(item is expected[index] for index, item in enumerate(ordered))
    assert [item["matrix"][3] for item in ordered] == [
        1.0,
        4.0,
        3.0,
        0.0,
        2.0,
        5.0,
    ]

    again = sort_by_material(
        [stone_a, grass_a, stone_b, metal, grass_b, stone_c]
    )
    assert [id(item) for item in again] == [id(item) for item in ordered]


def test_instance_index_checks_overflow():
    assert instance_index(100, 3, 12) == 136
    assert instance_index(MAX_INDEX - 11, 1, 12) == MAX_INDEX

    with pytest.raises(InstanceError):
        instance_index(MAX_INDEX - 10, 1, 12)


def test_dedupe_compares_matrices_with_epsilon():
    base_matrix = make_matrix()
    near_matrix = list(base_matrix)
    near_matrix[0] = 1.0 + 4e-7
    far_matrix = make_matrix(tx=0.01)

    first = make_instance(matrix=base_matrix)
    duplicate = make_instance(matrix=near_matrix)
    distinct = make_instance(matrix=far_matrix)

    deduped = dedupe_instances([first, duplicate, distinct], EPS)

    assert len(deduped) == 2
    assert deduped[0] is first
    assert deduped[1] is distinct


def test_validate_reports_singular_mesh_and_visibility():
    instances = [
        make_instance("grass", "mat-a", make_matrix(scale=0.0)),
        make_instance("rock", "mat-a", make_matrix(tx=1.0)),
        make_instance("rock", "mat-a", make_matrix(tx=2.0)),
        make_instance("moss", "mat-a", make_matrix(tx=3.0)),
        make_instance("grass", "mat-a", make_matrix(tx=4.0), visible=False),
    ]

    issues = validate_instances(instances, {"grass": 2, "rock": 1})

    assert issues == [
        "singular matrix",
        "mesh limit exceeded",
        "unknown mesh",
        "invisible instance",
    ]


def test_instance_batch_skips_invisible():
    frames = [
        [
            make_instance(matrix=make_matrix(tx=0.0), visible=True),
            make_instance(matrix=make_matrix(tx=1.0), visible=False),
            make_instance(matrix=make_matrix(tx=2.0), visible=True),
        ]
    ]

    result = instance_batch(frames, {"max_instances": 2})
    frame = result["frames"][0]

    assert frame["groups"] == {("grass", "mat-a"): [0, 2]}
    assert frame["batches"] == [[0, 2]]
    assert frame["draws"] == 1
    assert result["draws"] == 1


def test_instance_batch_isolates_frames():
    frames = [
        [
            make_instance("grass", "mat-a", make_matrix(tx=0.0)),
            make_instance("grass", "mat-a", make_matrix(tx=1.0)),
        ],
        [make_instance("rock", "mat-b", make_matrix(tx=2.0))],
        [make_instance("grass", "mat-a", make_matrix(tx=3.0))],
    ]

    result = instance_batch(frames, {"max_instances": 4})

    assert [list(frame["groups"]) for frame in result["frames"]] == [
        [("grass", "mat-a")],
        [("rock", "mat-b")],
        [("grass", "mat-a")],
    ]
    assert [frame["batches"] for frame in result["frames"]] == [
        [[0, 1]],
        [[0]],
        [[0]],
    ]
    assert [frame["draws"] for frame in result["frames"]] == [1, 1, 1]
    assert result["draws"] == 3

    broken = make_instance("grass", "mat-a", make_matrix(tx=4.0))
    broken["matrix"] = broken["matrix"][:15]
    with pytest.raises(InstanceError):
        instance_batch([frames[0], [broken]], {"max_instances": 4})


def test_instance_batch_rejects_zero_max_instances():
    frames = [[make_instance()]]

    with pytest.raises(InstanceError):
        instance_batch(frames, {"max_instances": 0})


def test_single_instance_forms_one_group():
    single = make_instance("grass", "mat-a", make_matrix(tx=0.0))

    groups = group_instances([single])

    assert list(groups) == [("grass", "mat-a")]
    assert groups[("grass", "mat-a")] == [single]


def test_exact_instance_limit_forms_one_batch():
    group = [
        make_instance(matrix=make_matrix(tx=0.0)),
        make_instance(matrix=make_matrix(tx=1.0)),
        make_instance(matrix=make_matrix(tx=2.0)),
    ]

    result = pack_matrices(group, max_instances=3)

    assert result == {
        "batches": [
            [
                make_matrix(tx=0.0),
                make_matrix(tx=1.0),
                make_matrix(tx=2.0),
            ]
        ]
    }


def test_empty_input_returns_empty_results():
    assert group_instances([]) == {}
    assert pack_matrices([], max_instances=4) == {"batches": []}
    assert cull_instances([], [(1.0, 0.0, 0.0, 0.0)]) == []
    assert sort_by_material([]) == []
    assert dedupe_instances([], EPS) == []
    assert validate_instances([], {}) == []
    assert instance_batch([], {"max_instances": 4}) == {
        "frames": [],
        "draws": 0,
    }


def test_pack_rejects_zero_max_instances():
    group = [make_instance()]

    with pytest.raises(InstanceError):
        pack_matrices(group, max_instances=0)


def test_instance_batch_preserves_success_order():
    rock = make_instance("rock", "mat-z", make_matrix(tx=0.0))
    grass_a = make_instance("grass", "mat-a", make_matrix(tx=1.0))
    grass_b = make_instance("grass", "mat-a", make_matrix(tx=2.0))
    second_frame = make_instance("grass", "mat-a", make_matrix(tx=3.0))

    result = instance_batch(
        [[rock, grass_a, grass_b], [second_frame]], {"max_instances": 4}
    )

    first = result["frames"][0]
    assert list(first["groups"]) == [
        ("rock", "mat-z"),
        ("grass", "mat-a"),
    ]
    assert first["groups"][("rock", "mat-z")] == [0]
    assert first["groups"][("grass", "mat-a")] == [1, 2]
    assert first["batches"] == [[0], [1, 2]]
    assert result["frames"][1]["batches"] == [[0]]
    assert [frame["draws"] for frame in result["frames"]] == [2, 1]
    assert result["draws"] == 3
