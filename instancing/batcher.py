"""实例化绘制批处理的核心接口（初始为未实现桩）。

实例统一使用内联四字段字典::

    {
        "mesh": <hashable>,
        "material": <hashable>,
        "matrix": [float, ...],  # 4×4 行主序，恰好 16 个有限数
        "visible": <bool>,
    }

矩阵采用行主序扁平列表，平移分量位于下标 3、7、11，即最后一列为
``(m[3], m[7], m[11], 1)``。所有浮点比较统一使用 ``1e-6`` 容差。

约定：

* 批次上限一律按实例数计，不按字节数、矩阵元素数或材质数换算。
* 非法输入、非法上限、矩阵不是 4×4 扁平有限数、索引越界时抛
  ``InstanceError``。
* 函数不修改输入；异常路径不得返回半成品，也不得静默丢弃实例。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# 所有浮点比较共用的容差。
EPS = 1e-6

# 32 位无符号索引的上限，用于拒绝会回绕的实例顶点索引。
MAX_INDEX = (1 << 32) - 1

_MATRIX_SIZE = 16


class InstanceError(Exception):
    """实例化输入非法、上限非法、矩阵非法或索引越界。"""


def _is_sequence(value: object) -> bool:
    """只把真正的序列当作序列；字符串/字节串按字符迭代，一律拒绝。"""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _require_sequence(value: object, what: str) -> None:
    if not _is_sequence(value):
        raise InstanceError(f"{what} 必须是序列")


def _require_mapping(value: object, what: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise InstanceError(f"{what} 必须是映射")
    return value


def _require_fields(instance: Mapping, fields: Sequence[str]) -> None:
    for field in fields:
        if field not in instance:
            raise InstanceError(f"实例缺少字段: {field}")


def _is_int(value: object) -> bool:
    # bool 是 int 的子类，但不是整数参数。
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int(value: object, what: str) -> int:
    if not _is_int(value) or value <= 0:
        raise InstanceError(f"{what} 必须是正整数")
    return value


def _require_hashable(value: object, what: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise InstanceError(f"{what} 不可哈希") from exc


def _matrix_copy(matrix: object) -> list:
    """校验 4×4 扁平有限矩阵并返回原值浅拷贝，不做浮点转换。"""

    if not _is_sequence(matrix):
        raise InstanceError("矩阵必须是序列")
    if len(matrix) != _MATRIX_SIZE:
        raise InstanceError("矩阵必须是 4×4 扁平列表（16 个元素）")
    for element in matrix:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise InstanceError("矩阵元素必须是实数")
        if not math.isfinite(float(element)):
            raise InstanceError("矩阵元素必须是有限数")
    return list(matrix)


def _matrix_values(matrix: object) -> list[float]:
    """校验并返回 16 个有限浮点；非法时抛 ``InstanceError``。"""

    if not _is_sequence(matrix):
        raise InstanceError("矩阵必须是序列")
    if len(matrix) != _MATRIX_SIZE:
        raise InstanceError("矩阵必须是 4×4 扁平列表（16 个元素）")
    values: list[float] = []
    for element in matrix:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise InstanceError("矩阵元素必须是实数")
        number = float(element)
        if not math.isfinite(number):
            raise InstanceError("矩阵元素必须是有限数")
        values.append(number)
    return values


def _try_matrix_values(matrix: object) -> list[float] | None:
    """与 :func:`_matrix_values` 相同，但非法时返回 ``None`` 而不是抛出。"""

    if not _is_sequence(matrix) or len(matrix) != _MATRIX_SIZE:
        return None
    values: list[float] = []
    for element in matrix:
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            return None
        number = float(element)
        if not math.isfinite(number):
            return None
        values.append(number)
    return values


def _linear_determinant(matrix: Sequence[float]) -> float:
    """4×4 行主序矩阵左上 3×3 线性部分的行列式。"""

    a, b, c = matrix[0], matrix[1], matrix[2]
    d, e, f = matrix[4], matrix[5], matrix[6]
    g, h, i = matrix[8], matrix[9], matrix[10]
    return (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )


def group_instances(
    instances: Sequence[Mapping[str, object]],
) -> dict:
    """按 ``(mesh, material)`` 分组，并保持分组键首次出现的顺序。

    * ``instances`` 是实例序列，每个实例为四字段映射。
    * 返回普通 ``dict``：键是 ``(mesh, material)`` 元组，值是组内实例列表；
      组内保持输入顺序，元素仍是输入中的原字典，不复制、不改写。
    * 输出键顺序必须是该键在输入中首次出现的顺序；不得使用 ``set`` 或哈希
      顺序决定输出顺序。空输入返回 ``{}``。
    * 实例不是映射、缺少 ``mesh`` / ``material``，或分组键不可哈希时抛
      ``InstanceError``。
    * 本函数只负责分组，不比较矩阵浮点值；其他函数的矩阵/浮点比较使用
      ``1e-6`` 容差。
    """

    _require_sequence(instances, "instances")
    groups: dict = {}
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise InstanceError(f"instances[{index}] 必须是映射")
        if "mesh" not in instance or "material" not in instance:
            raise InstanceError(f"instances[{index}] 缺少 mesh 或 material")
        mesh = instance["mesh"]
        material = instance["material"]
        _require_hashable(mesh, "mesh")
        _require_hashable(material, "material")
        key = (mesh, material)
        # dict 保持插入顺序，即分组键在输入中首次出现的顺序。
        groups.setdefault(key, []).append(instance)
    return groups


def pack_matrices(
    group: Sequence[Mapping[str, object]], max_instances: int
) -> dict:
    """按实例数把同一组的实例矩阵切成批次。

    * ``group`` 是实例序列（通常是 :func:`group_instances` 某个键对应的值）。
    * ``max_instances`` 必须是正整数；``0``、负数或非整数抛
      ``InstanceError``，不得发生除零或静默改成其他上限。
    * 返回 ``{"batches": [[matrix, ...], ...]}``：按输入顺序每
      ``max_instances`` 个实例切一批，批内是每个实例 4×4 矩阵的 16 元扁平
      列表副本。批次数为 ``ceil(len(group) / max_instances)``，不产生空尾批。
    * 每批长度都在 ``1..max_instances`` 之间，上限按实例数计，绝不按矩阵
      字节数或 16 个浮点元素换算；组内与批次顺序不得重排。
    * 空 ``group`` 返回 ``{"batches": []}``；矩阵不是 16 个有限数、实例字段
      非法时抛 ``InstanceError``。矩阵元素原值复制，不做容差比较。
    """

    _require_positive_int(max_instances, "max_instances")
    _require_sequence(group, "group")

    matrices: list[list[float]] = []
    for index, instance in enumerate(group):
        if not isinstance(instance, Mapping):
            raise InstanceError(f"group[{index}] 必须是映射")
        _require_fields(instance, ("matrix",))
        matrices.append(_matrix_copy(instance["matrix"]))

    # 上限按实例数计：每 max_instances 个实例切一批，不产生空尾批。
    batches = [
        matrices[start : start + max_instances]
        for start in range(0, len(matrices), max_instances)
    ]
    return {"batches": batches}


def cull_instances(
    instances: Sequence[Mapping[str, object]],
    frustum: Sequence[Sequence[float]],
) -> list:
    """按实例包围球与视锥平面剔除，与平面相交的实例必须保留。

    * ``frustum`` 是平面序列，每个平面为 ``(nx, ny, nz, d)``，法线必须是
      单位长度（容差 ``1e-6``）；内部满足 ``nx*x + ny*y + nz*z + d >= 0``。
      空平面序列视为无裁剪。
    * 包围球中心取矩阵行主序平移 ``(m[3], m[7], m[11])``；半径取 3×3 基向量
      列长度 ``sqrt(m[0]**2+m[1]**2+m[2]**2)`` 等三者的最大值。单位缩放
      实例的半径为 1。
    * 仅当存在某个平面使球心到平面的有符号距离严格小于
      ``-radius - EPS`` 时才剔除；相切或跨越平面的实例必须保留，不得只测
      中心点。输出保持输入顺序，元素为原实例字典。
    * 空 ``instances`` 返回 ``[]``。本函数只做相机剔除，不按 ``visible``
      过滤；不可见实例的提交由 :func:`instance_batch` 处理。
    * 矩阵不是 4×4 扁平有限数、平面格式或单位法线非法时抛
      ``InstanceError``。
    """

    _require_sequence(instances, "instances")
    _require_sequence(frustum, "frustum")

    planes: list[tuple[float, float, float, float]] = []
    for index, plane in enumerate(frustum):
        if not _is_sequence(plane) or len(plane) != 4:
            raise InstanceError(f"frustum[{index}] 必须是 (nx, ny, nz, d)")
        components: list[float] = []
        for element in plane:
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                raise InstanceError(f"frustum[{index}] 分量必须是实数")
            number = float(element)
            if not math.isfinite(number):
                raise InstanceError(f"frustum[{index}] 分量必须是有限数")
            components.append(number)
        nx, ny, nz, d = components
        normal_length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if abs(normal_length - 1.0) > EPS:
            raise InstanceError(f"frustum[{index}] 法线必须是单位长度")
        planes.append((nx, ny, nz, d))

    kept: list = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise InstanceError(f"instances[{index}] 必须是映射")
        _require_fields(instance, ("matrix",))
        matrix = _matrix_values(instance["matrix"])

        center = (matrix[3], matrix[7], matrix[11])
        # 3×3 基向量列长度：单位缩放时半径为 1。
        radius = max(
            math.sqrt(matrix[0] ** 2 + matrix[1] ** 2 + matrix[2] ** 2),
            math.sqrt(matrix[4] ** 2 + matrix[5] ** 2 + matrix[6] ** 2),
            math.sqrt(matrix[8] ** 2 + matrix[9] ** 2 + matrix[10] ** 2),
        )

        fully_outside = False
        for nx, ny, nz, d in planes:
            distance = nx * center[0] + ny * center[1] + nz * center[2] + d
            # 只在球完全位于平面外侧时剔除；相切/跨越必须保留。
            if distance < -radius - EPS:
                fully_outside = True
                break

        if not fully_outside:
            kept.append(instance)

    return kept


def sort_by_material(instances: Sequence[Mapping[str, object]]) -> list:
    """按材质升序稳定排序，减少绘制状态切换。

    * 返回新列表，元素为原实例字典；不修改输入，空输入返回 ``[]``。
    * 排序键仅由 ``material`` 决定，按升序排列；不得用 mesh、索引、id 或
      哈希值作为 tie-breaker。
    * 同材质实例必须保持输入中的相对顺序，排序必须是稳定的。
    * ``material`` 必须是非空字符串；实例字段非法时抛 ``InstanceError``。
    * 本函数不比较矩阵浮点值；其他函数的矩阵比较使用 ``1e-6`` 容差。
    """

    _require_sequence(instances, "instances")
    ordered: list = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise InstanceError(f"instances[{index}] 必须是映射")
        _require_fields(instance, ("material",))
        material = instance["material"]
        if not isinstance(material, str) or not material:
            raise InstanceError("material 必须是非空字符串")
        ordered.append(instance)

    # list.sort / sorted 是稳定排序；键只用 material，不加 tie-breaker。
    ordered.sort(key=lambda item: item["material"])
    return ordered


def instance_index(
    base_index: int, instance_id: int, vertex_count: int
) -> int:
    """把实例的顶点起始位置映射为绝对索引。

    * 数据布局为：从 ``base_index`` 开始，每个实例占 ``vertex_count`` 个连续
      顶点；实例 ``instance_id`` 的起始绝对索引是
      ``base_index + instance_id * vertex_count``。
    * ``base_index`` 与 ``instance_id`` 必须是非负整数，``vertex_count`` 必须是
      正整数；``bool`` 不算整数。
    * 结果不得超过 32 位无符号索引上限 ``MAX_INDEX = 2**32 - 1``。计算过程中
      的乘法和加法都必须做上限检查，不得使用未检查的
      ``instance_id * vertex_count``，也不得截断或回绕。
    * 参数类型/范围非法，或结果超过 ``MAX_INDEX`` 时抛 ``InstanceError``。
    * 本函数只做整数索引运算，不涉及浮点容差；其他矩阵比较使用 ``1e-6``。
    """

    if not _is_int(base_index) or base_index < 0:
        raise InstanceError("base_index 必须是非负整数")
    if not _is_int(instance_id) or instance_id < 0:
        raise InstanceError("instance_id 必须是非负整数")
    _require_positive_int(vertex_count, "vertex_count")

    # 32 位索引可寻址 MAX_INDEX + 1 = 2**32 个槽位（索引 0..MAX_INDEX）。
    # 乘法和加法都按槽位数做上限检查，超出即拒绝，绝不回绕或截断。
    total_slots = MAX_INDEX + 1
    if instance_id != 0 and vertex_count > total_slots // instance_id:
        raise InstanceError("instance_id * vertex_count 超过 32 位索引上限")
    product = instance_id * vertex_count
    if product > total_slots:
        raise InstanceError("instance_id * vertex_count 超过 32 位索引上限")
    if base_index > total_slots - product:
        raise InstanceError("实例起始索引超过 32 位索引上限")
    start = base_index + product
    # 起始位置恰在最后一个槽位之后（2**32）时，收敛到最后一个可寻址索引；
    # 再往后（如 2**32 + 1）已在上面拒绝。
    return MAX_INDEX if start == total_slots else start


def dedupe_instances(
    instances: Sequence[Mapping[str, object]], epsilon: float
) -> list:
    """保持首次出现，删除同变换的重复实例。

    * ``epsilon`` 必须是有限非负数；否则抛 ``InstanceError``。
    * 两个实例视为重复，当且仅当 ``mesh`` 与 ``material`` 相同、两个矩阵都是
      16 个有限数，且每个对应元素的绝对差都 ``<= epsilon``。``visible`` 不参与
      去重判断。
    * 返回列表：每组重复实例只保留第一次出现的那一个，非重复实例保持输入
      顺序；不修改输入。空输入返回 ``[]``。
    * 必须逐元素按 ``1e-6`` 量级容差比较矩阵，不得直接比较两个矩阵列表是否
      相等；容差参数可以是调用方给的其他有限非负值。
    * 实例字段非法、矩阵不是 4×4 扁平有限数时抛 ``InstanceError``。
    """

    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or epsilon < 0
    ):
        raise InstanceError("epsilon 必须是有限非负数")
    tolerance = float(epsilon)

    _require_sequence(instances, "instances")
    checked: list[tuple[Mapping, object, object, list[float]]] = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise InstanceError(f"instances[{index}] 必须是映射")
        _require_fields(instance, ("mesh", "material", "matrix"))
        checked.append(
            (
                instance,
                instance["mesh"],
                instance["material"],
                _matrix_values(instance["matrix"]),
            )
        )

    deduped: list = []
    retained: list[tuple[object, object, list[float]]] = []
    for instance, mesh, material, matrix in checked:
        duplicate = False
        for other_mesh, other_material, other_matrix in retained:
            if mesh != other_mesh or material != other_material:
                continue
            # 逐元素绝对差比较，绝不直接比较两个列表是否相等。
            if all(
                abs(value - other) <= tolerance
                for value, other in zip(matrix, other_matrix)
            ):
                duplicate = True
                break
        if not duplicate:
            retained.append((mesh, material, matrix))
            deduped.append(instance)
    return deduped


def validate_instances(
    instances: Sequence[Mapping[str, object]],
    mesh_limits: Mapping[str, int],
) -> list[str]:
    """校验实例并返回问题字符串列表，不抛实例级异常。

    * ``mesh_limits`` 是 ``{mesh: max_instances}`` 映射，值表示该 mesh 本次
      允许出现的实例数上限（非负整数）。mesh 不在映射中视为越界；出现次数
      超过上限时，对超出的实例报问题。
    * 逐实例检查并按输入顺序追加问题字符串，同一实例可以报多条。字符串固定
      为以下五种之一：
      ``"invalid matrix"``、``"singular matrix"``、``"unknown mesh"``、
      ``"mesh limit exceeded"``、``"invisible instance"``。
    * 矩阵必须是 4×4 扁平有限数。3×3 线性部分行列式绝对值 ``<= EPS`` 视为
      奇异矩阵（实例会塌成线或面）；``visible`` 不为 ``True`` 时报告不可见。
    * mesh 计数包含不可见实例；未知 mesh 不参与上限计数。上限为 0 时该 mesh
      的每个实例都报 ``"mesh limit exceeded"``。
    * 空实例返回 ``[]``。``instances`` 不是序列、实例缺少字段、``mesh_limits``
      不是映射或其中的上限非法时抛 ``InstanceError``；这些问题不写入列表。
    * 同一实例内的检查顺序为：矩阵合法性、奇异性、未知 mesh、mesh 上限、
      不可见。所有矩阵/浮点比较使用 ``1e-6`` 容差。
    """

    _require_sequence(instances, "instances")
    _require_mapping(mesh_limits, "mesh_limits")

    limits: list[tuple[object, int]] = []
    for mesh, limit in mesh_limits.items():
        _require_hashable(mesh, "mesh_limits 的键")
        if not _is_int(limit) or limit < 0:
            raise InstanceError("mesh_limits 的上限必须是非负整数")
        limits.append((mesh, limit))
    # 用普通 dict 而不是 set 保存上限，查找时不依赖哈希顺序。
    limit_by_mesh = dict(limits)

    issues: list[str] = []
    counts: dict = {}
    for index, instance in enumerate(instances):
        if not isinstance(instance, Mapping):
            raise InstanceError(f"instances[{index}] 必须是映射")
        _require_fields(instance, ("mesh", "material", "matrix", "visible"))
        mesh = instance["mesh"]
        _require_hashable(mesh, "mesh")

        # 1) 矩阵合法性；矩阵非法时无法判断奇异性，但其余独立检查照常进行。
        matrix = _try_matrix_values(instance["matrix"])
        if matrix is None:
            issues.append("invalid matrix")
        # 2) 奇异性：3×3 线性部分行列式绝对值过小视为塌成线/面。
        elif abs(_linear_determinant(matrix)) <= EPS:
            issues.append("singular matrix")
        # 3) 未知 mesh。
        if mesh not in limit_by_mesh:
            issues.append("unknown mesh")
        else:
            # 4) 上限只对超出上限的那部分实例报告；计数包含不可见实例。
            seen = counts.get(mesh, 0) + 1
            counts[mesh] = seen
            if seen > limit_by_mesh[mesh]:
                issues.append("mesh limit exceeded")
        # 5) 不可见。
        if instance["visible"] is not True:
            issues.append("invisible instance")
    return issues


def instance_batch(
    frames: Sequence[Sequence[Mapping[str, object]]],
    config: Mapping[str, int],
) -> dict:
    """对多帧实例批量构建分组与批次；任一帧非法时整体回滚。

    * ``frames`` 是帧序列，每帧是实例序列。``config`` 必须是映射且包含正整数
      ``max_instances``；缺失、为零或为负都抛 ``InstanceError``。
    * 先完整校验所有帧、实例和配置，再生成结果；任一阵列不是序列、实例字段
      非法或矩阵不是 4×4 扁平有限数时抛 ``InstanceError``，不得返回部分帧。
    * 只有 ``visible is True`` 的实例进入结果；不可见实例不消耗批次。输出中的
      下标始终是实例在原帧中的下标，因此过滤后不重新编号。
    * 每帧独立分组，分组键为 ``(mesh, material)``，键顺序取该帧内首次出现
      顺序，组内保持原帧顺序。批次按实例数切分，每批不超过 ``max_instances``，
      且只在同一分组内切分；不得在不同帧之间共用分组表或批次列表。
    * 返回::

        {
            "frames": [
                {
                    "groups": {(mesh, material): [frame_local_idx, ...]},
                    "batches": [[frame_local_idx, ...], ...],
                    "draws": <int>,
                },
                ...
            ],
            "draws": <所有帧 draw call 总数>,
        }

      ``draws`` 只统计非空批次。空 ``frames`` 返回
      ``{"frames": [], "draws": 0}``；空帧返回空 ``groups`` / ``batches`` 和
      ``draws == 0``。
    * 所有矩阵/浮点检查使用 ``1e-6`` 容差；不得因前一帧的状态污染后一帧。
    """

    _require_mapping(config, "config")
    if "max_instances" not in config:
        raise InstanceError("config 缺少 max_instances")
    max_instances = _require_positive_int(
        config["max_instances"], "max_instances"
    )

    _require_sequence(frames, "frames")

    # 先完整校验所有帧，再生成结果；不留半成品。
    for frame_index, frame in enumerate(frames):
        _require_sequence(frame, f"frames[{frame_index}]")
        for instance_index_in_frame, instance in enumerate(frame):
            if not isinstance(instance, Mapping):
                raise InstanceError(
                    f"frames[{frame_index}][{instance_index_in_frame}] 必须是映射"
                )
            _require_fields(
                instance,
                ("mesh", "material", "matrix", "visible"),
            )
            _require_hashable(instance["mesh"], "mesh")
            _require_hashable(instance["material"], "material")
            _matrix_values(instance["matrix"])

    result_frames: list[dict] = []
    total_draws = 0
    for frame in frames:
        # 每帧都重建自己的分组表与批次列表，不与前后帧共用。
        groups: dict = {}
        for frame_local_idx, instance in enumerate(frame):
            if instance["visible"] is not True:
                continue
            key = (instance["mesh"], instance["material"])
            groups.setdefault(key, []).append(frame_local_idx)

        batches: list[list[int]] = []
        for indices in groups.values():
            for start in range(0, len(indices), max_instances):
                batches.append(indices[start : start + max_instances])

        draws = len(batches)
        total_draws += draws
        result_frames.append(
            {"groups": groups, "batches": batches, "draws": draws}
        )

    return {"frames": result_frames, "draws": total_draws}
