"""
    Regenerates main_window_ui.py from window.ui
    pyuic5 must be installed and in your PATH
"""

import subprocess

subprocess.run(["pyuic5", "-x", "window.ui", "-o", "main_window_ui.py"], text=True, check=True)
f = open("main_window_ui.py", 'r')
lines = f.read().split('\n')
f.close()

graph = []
lines_out = []

for line in lines:
    if "self.plotter" in line:
        if "Plotter" in line:
            graph.append("        self.plotter = Plotter(self.centralwidget)")
        else:
            graph.append(line)
    elif "self.toolbar = NavigationToolbar2QT(MainWindow)" in line:
        lines_out.append("        self.toolbar = NavigationToolbar2QT(self.plotter, MainWindow)")
    elif "MainWindow.setCentralWidget(self.centralwidget)" in line:
        for g_line in graph:
            lines_out.append(g_line)
        lines_out.append(line)
    else:
        lines_out.append(line)

f = open("main_window_ui.py", 'w')
f.write('\n'.join(lines_out))
f.close()
