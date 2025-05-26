# live_plot.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from PIL import Image
import io

class MultiLivePlot:
    def __init__(self, titles, ncols=3):
        self.num_plots = len(titles)
        self.ncols = ncols
        self.nrows = math.ceil(self.num_plots / self.ncols)

        self.fig, self.axes = plt.subplots(self.nrows, self.ncols, figsize=(5 * self.ncols, 4 * self.nrows))
        self.axes = self.axes.flatten()
        self.lines = []
        self.data = [[] for _ in range(self.num_plots)]
        self.steps = []
        self.titles = titles
        self.frames = []  # ⬅️ 记录每帧图像

        for i in range(self.num_plots):
            ax = self.axes[i]
            line, = ax.plot([], [], label=titles[i])
            ax.set_title(titles[i])
            ax.set_xlabel('Step')
            ax.set_ylabel(titles[i])
            ax.legend()
            self.lines.append(line)

        for j in range(self.num_plots, len(self.axes)):
            self.axes[j].set_visible(False)

        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, step, values):
        self.steps.append(step)
        for i in range(self.num_plots):
            self.data[i].append(values[i])
            self.lines[i].set_data(self.steps, self.data[i])
            self.axes[i].relim()
            self.axes[i].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 每帧保存为图像（Pillow格式）
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png')
        buf.seek(0)
        self.frames.append(Image.open(buf))

    def save_gif(self, filename='multi_plot.gif', duration=200):
        if self.frames:
            self.frames[0].save(
                filename,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0
            )
