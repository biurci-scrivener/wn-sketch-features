import sys
from PyQt5.QtWidgets import (
	QApplication, QMainWindow, QFileDialog, QColorDialog
)
from main_window_ui import Ui_MainWindow
import os

class Window(QMainWindow, Ui_MainWindow):

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setupUi(self)
		self.connect()

	def connect(self):
		self.button_browse.clicked.connect(self.mybrowse)
		self.button_load.clicked.connect(self.myload)
		self.button_clear.clicked.connect(self.myclear)
		self.button_color.clicked.connect(self.mycolor)
		self.button_auto.clicked.connect(self.myauto)
		self.button_save.clicked.connect(self.mysave)

		self.slider_strength.valueChanged.connect(self.plotter.on_update_slider)

	def mybrowse(self):
		options = QFileDialog.Options()
		path, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*)", options=options)
		if path:
			self.box_path.setText(path)

	def myload(self):
		if os.path.isfile(self.box_path.text()):
			self.plotter.load_sketch(self.box_path.text())
			print("Loaded sketch")
		else:
			print("Bad filename")
	
	def myclear(self):
		self.plotter.reset_plot_attributes(sketch=True)
	
	def mycolor(self):
		self.plotter.pick_color()

	def myauto(self):
		self.plotter.gen_auto_seeds()

	def mysave(self):
		self.plotter.save_png()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	win = Window()
	win.show()
	sys.exit(app.exec())