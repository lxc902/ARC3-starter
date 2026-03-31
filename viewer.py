"""
Live viewer for StochasticGoose agent.
Run from your own terminal:  python viewer.py

Controls:
  - Slider: scrub to any step
  - Left/Right arrow keys: step backward/forward
  - Space: toggle auto-follow (live tail vs. manual scrub)
"""
import os
import sys
import glob
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

ARC_COLORS = {
    0: (1.0, 1.0, 1.0),       1: (0.8, 0.8, 0.8),
    2: (0.6, 0.6, 0.6),       3: (0.4, 0.4, 0.4),
    4: (0.2, 0.2, 0.2),       5: (0.0, 0.0, 0.0),
    6: (0.898, 0.227, 0.639), 7: (1.0, 0.482, 0.8),
    8: (0.976, 0.235, 0.192), 9: (0.118, 0.576, 1.0),
    10: (0.533, 0.847, 0.945), 11: (1.0, 0.863, 0.0),
    12: (1.0, 0.522, 0.106),  13: (0.573, 0.071, 0.192),
    14: (0.310, 0.800, 0.188), 15: (0.639, 0.337, 0.839),
}
ACTION_NAMES = ["Up", "Down", "Left", "Right", "Space", "Click"]
arc_cmap = ListedColormap([ARC_COLORS[i] for i in range(16)])


def find_viewer_dir(base='runs'):
    for rd in sorted(glob.glob(os.path.join(base, '*')), reverse=True):
        for gd in glob.glob(os.path.join(rd, '*')):
            vd = os.path.join(gd, 'viewer_data')
            if os.path.isdir(vd) and os.path.exists(os.path.join(vd, 'steps.jsonl')):
                return vd
    return None


class LiveViewer:
    def __init__(self, viewer_dir):
        self.viewer_dir = viewer_dir
        self.steps_path = os.path.join(viewer_dir, 'steps.jsonl')
        self.heatmaps_dir = os.path.join(viewer_dir, 'heatmaps')
        self.steps = []
        self.current_idx = 0
        self.auto_follow = True
        self.last_file_pos = 0
        self._load_steps()

        self.fig = plt.figure(figsize=(16, 9), facecolor='#1a1a2e')
        self.fig.canvas.manager.set_window_title('StochasticGoose Live Viewer')

        gs = gridspec.GridSpec(3, 2, height_ratios=[5, 1.2, 0.5],
                               width_ratios=[1, 1], hspace=0.35, wspace=0.25,
                               left=0.06, right=0.96, top=0.93, bottom=0.08)

        self.ax_grid = self.fig.add_subplot(gs[0, 0])
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_bars = self.fig.add_subplot(gs[1, :])
        self.ax_slider = self.fig.add_subplot(gs[2, :])
        for ax in (self.ax_grid, self.ax_heat, self.ax_bars):
            ax.set_facecolor('#16213e')

        self.title = self.fig.suptitle('', fontsize=13, color='white', fontweight='bold')

        maxs = max(len(self.steps) - 1, 1)
        self.slider = Slider(self.ax_slider, 'Step', 0, maxs, valinit=0,
                             valstep=1, color='#4A90D9')
        self.slider.label.set_color('white')
        self.slider.valtext.set_color('white')
        self.slider.on_changed(self._on_slider_changed)

        btn_ax = self.fig.add_axes([0.88, 0.01, 0.09, 0.035])
        self.btn = Button(btn_ax, 'LIVE', color='#2ECC71', hovercolor='#27AE60')
        self.btn.on_clicked(self._toggle_follow)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._im_grid = None
        self._im_heat = None
        self._render()

    # ---- data loading ----
    def _load_steps(self):
        if not os.path.exists(self.steps_path):
            return
        with open(self.steps_path, 'r') as f:
            f.seek(self.last_file_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.steps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            self.last_file_pos = f.tell()

    def _load_heatmap(self, step_num):
        p = os.path.join(self.heatmaps_dir, f'{step_num:06d}.npy')
        return np.load(p).astype(np.float32) if os.path.exists(p) else None

    # ---- rendering ----
    def _render(self):
        if not self.steps:
            return
        idx = min(self.current_idx, len(self.steps) - 1)
        s = self.steps[idx]
        frame = np.array(s['frame'])
        ap = np.array(s['action_probs'])
        ai = s['action_idx']
        cx, cy = s['coord_x'], s['coord_y']
        hm = self._load_heatmap(s['step'])

        act_name = ACTION_NAMES[ai] if ai < 5 else f"Click({cx},{cy})"
        mode = "LIVE" if self.auto_follow else "REVIEW"
        self.title.set_text(
            f'[{mode}]  Step {s["step"]}  |  {act_name}  |  '
            f'Levels: {s["levels_completed"]}  |  Buffer: {s["buffer_size"]}  |  '
            f'Loaded: {len(self.steps)}'
        )

        # --- game grid ---
        self.ax_grid.clear()
        self.ax_grid.imshow(frame, cmap=arc_cmap, vmin=0, vmax=15,
                            interpolation='nearest', aspect='equal')
        if ai >= 5 and cx >= 0:
            self.ax_grid.plot(cx, cy, 'rx', ms=10, mew=2.5)
        self.ax_grid.set_title('Game Grid', color='white', fontsize=11)
        self.ax_grid.tick_params(colors='#666', labelsize=7)

        # --- heatmap ---
        self.ax_heat.clear()
        if hm is not None:
            self.ax_heat.imshow(hm, cmap='hot', interpolation='bilinear',
                                aspect='equal', vmin=0, vmax=max(hm.max(), 0.01))
            if ai >= 5 and cx >= 0:
                self.ax_heat.plot(cx, cy, 'c+', ms=12, mew=2)
            self.ax_heat.set_title(f'Click Heatmap (max:{hm.max():.3f})',
                                   color='white', fontsize=11)
        else:
            self.ax_heat.set_title('Click Heatmap (waiting...)', color='white', fontsize=11)
        self.ax_heat.tick_params(colors='#666', labelsize=7)

        # --- bar chart ---
        self.ax_bars.clear()
        click_p = float(hm.mean()) if hm is not None else 0.0
        probs = list(ap) + [click_p]
        sel = ai if ai < 5 else 5
        colors = ['#2ECC71' if i == sel else '#4A90D9' for i in range(5)]
        colors.append('#E74C3C' if sel == 5 else '#D94A4A')
        bars = self.ax_bars.barh(range(6), probs, color=colors, height=0.6)
        self.ax_bars.set_yticks(range(6))
        self.ax_bars.set_yticklabels(ACTION_NAMES, color='white', fontsize=10)
        self.ax_bars.invert_yaxis()
        self.ax_bars.set_xlim(0, max(max(probs) * 1.3, 0.1))
        self.ax_bars.set_title('Action Probabilities (green=selected)', color='white', fontsize=11)
        self.ax_bars.tick_params(colors='#666', labelsize=8)
        for bar, p in zip(bars, probs):
            self.ax_bars.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                              f'{p:.4f}', va='center', color='white', fontsize=9)

    # ---- callbacks ----
    def _on_slider_changed(self, val):
        self.current_idx = int(val)
        self.auto_follow = False
        self.btn.label.set_text('REVIEW')
        self.btn.color = '#E74C3C'
        self._render()

    def _toggle_follow(self, _event):
        self.auto_follow = not self.auto_follow
        self.btn.label.set_text('LIVE' if self.auto_follow else 'REVIEW')
        self.btn.color = '#2ECC71' if self.auto_follow else '#E74C3C'
        if self.auto_follow and self.steps:
            self.current_idx = len(self.steps) - 1
            self.slider.set_val(self.current_idx)

    def _on_key(self, event):
        if event.key == 'right' and self.current_idx < len(self.steps) - 1:
            self.auto_follow = False
            self.current_idx += 1
            self.slider.set_val(self.current_idx)
        elif event.key == 'left' and self.current_idx > 0:
            self.auto_follow = False
            self.current_idx -= 1
            self.slider.set_val(self.current_idx)
        elif event.key == ' ':
            self._toggle_follow(None)

    def _update(self, _frame_num):
        old_n = len(self.steps)
        self._load_steps()
        if len(self.steps) != old_n:
            self.slider.valmax = max(len(self.steps) - 1, 1)
            self.slider.ax.set_xlim(0, self.slider.valmax)
            if self.auto_follow:
                self.current_idx = len(self.steps) - 1
                self.slider.set_val(self.current_idx)
                self._render()
        return []

    def run(self):
        self.anim = FuncAnimation(self.fig, self._update, interval=500,
                                  blit=False, cache_frame_data=False)
        plt.show()


def main():
    vd = sys.argv[1] if len(sys.argv) > 1 else find_viewer_dir()
    if not vd or not os.path.isdir(vd):
        print("No viewer_data found. Start the agent first.")
        print("Usage: python viewer.py [path/to/viewer_data]")
        sys.exit(1)
    print(f"Connected: {vd}", flush=True)
    print("Keys: ←/→ step | Space toggle live/review | Slider scrub", flush=True)
    LiveViewer(vd).run()


if __name__ == '__main__':
    main()
