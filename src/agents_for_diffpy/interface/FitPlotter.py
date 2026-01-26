from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys
from agents_for_diffpy.interface import FitRunner


class FitPlotter:
    def __init__(self):
        super().__init__()
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("Realtime Plot")
        self.layout = QtWidgets.QVBoxLayout()
        self.win.setLayout(self.layout)
        self.win.runner = None
        self.curves = []

    def connect_to_runner(self, runner: FitRunner):
        self.runner = runner
        for i, (window_id, data_pack) in enumerate(
            runner.data_for_plot.items()
        ):
            plot = pg.PlotWidget(title=data_pack["title"])
            self.layout.addWidget(plot)
            if data_pack["style"] == "sparse":
                curve = plot.plot(
                    pen=pg.mkPen(pg.intColor(i), width=2),
                    symbol="o",  # round marker
                    symbolSize=10,  # marker size
                    symbolBrush=pg.intColor(i),  # color of marker
                )
            else:
                curve = plot.plot(
                    pen=pg.mkPen(pg.intColor(i), width=2),
                )
            self.curves.append(curve)
        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 20Hz
        self.buffers = [[] for _ in range(len(self.curves))]

    def update_plot(self):
        for i, (window_id, data_pack) in enumerate(
            self.runner.data_for_plot.items()
        ):
            if not data_pack["ydata"].empty():
                new_data = data_pack["ydata"].get()
                if data_pack["update_mode"] == "append":
                    self.buffers[i].append(new_data)
                    plot_data = self.buffers[i]
                elif data_pack["update_mode"] == "replace":
                    plot_data = new_data
                self.curves[i].setData(plot_data)

    def on(self):
        self.win.show()
        sys.exit(self.app.exec_())
