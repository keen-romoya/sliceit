"""Force HUD strip composited under rollout video frames.

Draws, from per-step telemetry (never pixels):
- knife-force-vs-time area line, accumulating left to right over the episode,
  with the current sample emphasized;
- a hero readout of the instantaneous force;
- a cut-completion progress bar and blade-depth readout;
- episode-reset markers.

Single series: no legend, the title names it. Completion is a different
scale from force, so it is a separate bar — never a second y-axis.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# strip palette (light surface; ink for text, one warm series hue for force)
SURFACE = "#FFFFFF"
INK = "#1F2933"
MUTED = "#616E7C"
GRID = "#E4E7EB"
FORCE = "#B23A25"       # warm: material pushing back
FORCE_FILL = "#B23A25"  # used at low alpha
COMPLETE = "#2F7D4F"


class ForceHud:
    def __init__(self, width_px, total_frames, fps=30.0, height_px=200, dpi=100):
        self.w = width_px / dpi
        self.h = height_px / dpi
        self.dpi = dpi
        self.fps = fps
        self.total_frames = total_frames
        self.t = []
        self.force = []
        self.resets = []

    def add(self, frame_idx, force_n, was_reset=False):
        self.t.append(frame_idx / self.fps)
        self.force.append(force_n)
        if was_reset:
            self.resets.append(frame_idx / self.fps)

    def render(self, completion, depth_mm):
        fig = plt.figure(figsize=(self.w, self.h), dpi=self.dpi)
        fig.patch.set_facecolor(SURFACE)

        # hero readout block (left 22%)
        ax_txt = fig.add_axes([0.0, 0.0, 0.21, 1.0])
        ax_txt.set_axis_off()
        ax_txt.text(0.12, 0.78, "KNIFE FORCE", fontsize=8, color=MUTED,
                    fontweight="bold", family="DejaVu Sans")
        ax_txt.text(0.12, 0.40, f"{self.force[-1]:5.1f} N", fontsize=22,
                    color=INK, fontweight="bold", family="DejaVu Sans")
        ax_txt.text(0.12, 0.12, f"blade depth {max(depth_mm, 0):.0f} mm",
                    fontsize=8, color=MUTED, family="DejaVu Sans")

        # completion bar (under the plot, right block)
        ax_bar = fig.add_axes([0.30, 0.06, 0.62, 0.07])
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_axis_off()
        ax_bar.barh(0.5, 1.0, height=1.0, color=GRID)
        ax_bar.barh(0.5, min(max(completion, 0.0), 1.0), height=1.0, color=COMPLETE)
        ax_bar.text(1.01, 0.5, f"cut {completion * 100:3.0f}%", fontsize=8,
                    color=MUTED, va="center", family="DejaVu Sans")

        # force-vs-time line (right block)
        ax = fig.add_axes([0.30, 0.30, 0.62, 0.58])
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlim(0, self.total_frames / self.fps)
        ymax = max(2.0, max(self.force) * 1.25)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for r in self.resets:
            ax.axvline(r, color=GRID, linewidth=1.0, linestyle=(0, (3, 3)))
        t = np.asarray(self.t)
        f = np.asarray(self.force)
        ax.fill_between(t, f, color=FORCE_FILL, alpha=0.14, linewidth=0)
        ax.plot(t, f, color=FORCE, linewidth=2.0)
        ax.plot(t[-1], f[-1], "o", color=FORCE, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlabel("time [s]", fontsize=7, color=MUTED, labelpad=1)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return img
