"""Per-frame geometry assertions for rendered slicing scenes.

Everything here is computed from simulation state and scene config — never
from pixels. The goal is to catch the class of error where the rendering
shows something the dynamics did not simulate (blade passing through the
cutting board, food looking pre-cut while the sim says it is whole, the
blade detaching from the arm, the food drifting off the board).

Usage:
    checker = SceneChecker(cfg)
    checker.begin_frame(frame_idx)
    checker.blade_vs_board(blade_box)
    checker.blade_attachment(blade_top_z, wrist_z)
    checker.mesh_on_board(mesh_points)          # bridge mode
    checker.seam_integrity(mesh_points, weld_groups, contact_started)
    ...
    print(checker.summary())   # e.g. "GEOMETRY: 0 violations in 4 checks over 220 frames"

Each check records violations with the frame index and a numeric magnitude,
so a failing render points straight at the offending frames.
"""
from collections import defaultdict

EPS = 1e-4


class SceneChecker:
    def __init__(self, cfg):
        self.cfg = cfg
        board = cfg.board_pos
        self.board_box = (board[0] - 0.175, board[0] + 0.175,
                          board[1] - 0.125, board[1] + 0.125,
                          board[2] - 0.01, board[2] + 0.01)
        self.board_top = board[2] + 0.01
        self.violations = defaultdict(list)
        self.checks_run = defaultdict(int)
        self.frames = 0
        self._frame = 0

    def begin_frame(self, frame_idx):
        self._frame = frame_idx
        self.frames += 1

    def _record(self, name, magnitude):
        self.violations[name].append((self._frame, magnitude))

    # ------------------------------------------------------------ the checks

    def blade_vs_board(self, blade_box):
        """The board is rigid and uncut in the dynamics: the rendered blade
        must never dip below its top surface while over it."""
        self.checks_run["blade_vs_board"] += 1
        x0, x1, y0, y1, z0, _ = blade_box
        over_board = (x1 > self.board_box[0] and x0 < self.board_box[1]
                      and y1 > self.board_box[2] and y0 < self.board_box[3])
        if over_board and z0 < self.board_top - EPS:
            self._record("blade_vs_board", self.board_top - z0)

    def blade_attachment(self, blade_top_z, wrist_z, max_gap=0.12):
        """The blade must hang from the arm, not float: its top edge stays
        within max_gap of the wrist origin and never above it."""
        self.checks_run["blade_attachment"] += 1
        gap = wrist_z - blade_top_z
        if gap < -EPS or gap > max_gap:
            self._record("blade_attachment", gap)

    def mesh_on_board(self, mesh_z_min, mesh_xy_center, tol=0.005):
        """The food rests on the board: its lowest point stays at board-top
        level (no floating, no sinking) and its center stays over the board."""
        self.checks_run["mesh_on_board"] += 1
        if abs(mesh_z_min - self.board_top) > tol:
            self._record("mesh_on_board", mesh_z_min - self.board_top)
        cx, cy = mesh_xy_center
        if not (self.board_box[0] < cx < self.board_box[1]
                and self.board_box[2] < cy < self.board_box[3]):
            self._record("mesh_on_board", float("nan"))

    def seam_integrity(self, weld_spread, contact_started, tol=1.5e-3):
        """Before the blade has ever engaged the material, the sim says the
        object is whole — so every duplicated (virtual-node) vertex pair must
        be co-located. A spread before contact means the render shows a cut
        that has not happened."""
        self.checks_run["seam_integrity"] += 1
        if not contact_started and weld_spread > tol:
            self._record("seam_integrity", weld_spread)

    def watertight(self, tris, points_z, bottom_tol=0.006):
        """One-time topology audit: boundary edges (used by a single
        triangle) may exist only along the resting bottom of the mesh. A
        boundary above that is a hole — rendered as a dark seam that reads
        as a pre-existing cut."""
        from collections import Counter
        self.checks_run["watertight"] += 1
        edges = Counter()
        for a, b, c in tris:
            for e in ((a, b), (b, c), (c, a)):
                edges[min(e), max(e)] += 1
        z_min = min(points_z)
        bad = [e for e, cnt in edges.items() if cnt == 1
               and max(points_z[e[0]], points_z[e[1]]) > z_min + bottom_tol]
        if bad:
            self._record("watertight", float(len(bad)))
        return len(bad)

    def bridge_tracking(self, tracking_err_m, tol=0.005):
        """Co-simulation coherence: the DiSECt knife must be where the Isaac
        blade is. A divergence means the rendered blade and the force-
        producing knife are different objects (the failure mode where the
        cut looks effortless because the FEM knife never touched anything)."""
        self.checks_run["bridge_tracking"] += 1
        if tracking_err_m > tol:
            self._record("bridge_tracking", tracking_err_m)

    def blade_speed_plausible(self, blade_z, prev_blade_z, dt, max_cmd=0.04, margin=3.0):
        """The blade may never move faster than the controller can command
        (within a tracking margin). A violation means an unmodeled force is
        driving the arm — the failure mode where a wrong-frame or wrong-sign
        force accelerates the blade instead of resisting it."""
        self.checks_run["blade_speed_plausible"] += 1
        if prev_blade_z is None:
            return
        speed = abs(blade_z - prev_blade_z) / dt
        if speed > max_cmd * margin:
            self._record("blade_speed_plausible", speed)

    def solid_clip(self, clips):
        """Proxy-geometry mode: rendered blade must not interpenetrate solid
        (uncut) material boxes."""
        self.checks_run["solid_clip"] += 1
        if clips:
            self._record("solid_clip", 1.0)

    # ------------------------------------------------------------- reporting

    def rows(self):
        for name, hits in self.violations.items():
            for frame, mag in hits:
                yield (frame, name, mag)

    def summary(self):
        total = sum(len(v) for v in self.violations.values())
        lines = [f"GEOMETRY: {total} violation(s) in "
                 f"{len(self.checks_run)} check(s) over {self.frames} frames"]
        for name, hits in sorted(self.violations.items()):
            worst = max(abs(m) for _, m in hits if m == m) if hits else 0.0
            first = hits[0][0]
            lines.append(f"  {name}: {len(hits)} frame(s), first at {first}, "
                         f"worst {worst * 1000:.1f} mm")
        return "\n".join(lines)
