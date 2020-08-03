# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to perform ...
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow, QShortcut, QSpinBox, QGroupBox, QSizePolicy, QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvas


import sys
import biasd as b
import numpy as np

from .smd_loader import ui_loader

class posterior(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()

	def initialize(self):
		self.vbox = QVBoxLayout()
		
		hbox = QHBoxLayout()
		bcorner = QPushButton("&Choose Posterior")
		bcorner.clicked.connect(self.load_posterior)
		bsave = QPushButton("&Save Figure")
		bsave.clicked.connect(self.savefig)

		hbox.addWidget(bcorner)
		hbox.addWidget(bsave)
		hbox.addStretch(1)
		self.vbox.addLayout(hbox)
		self.vbox.addStretch(1)
		
		self.setLayout(self.vbox)
		self.setMinimumSize(800,800)
		self.setGeometry(200,0,800,800)
		self.setWindowTitle('View Posteriors')
		self.show()
	
	def savefig(self):
		oname = QFileDialog.getSaveFileName(self,"Save Figure",'./','jpg (*.jpg);;png (*.png);;pdf (*.pdf);;eps (*.eps)')
		self.setFocus()
		try:
			self.fig.print_figure(oname[0]) # B/c it's a canvas not a figure....
		except:
			QMessageBox.critical(None,"Could Not Save","Could not save file: %s\n."%(oname[0]))
	
	def get_smd_filename(self):
		return self.parent().parent().get_smd_filename()
	
	def load_posterior(self):
		try:
			self.loader.close()
		except:
			pass
		self.loader = ui_loader(self,select=True)
		self.loader.show()
		
	def select_callback(self,location):
		self.loader.close()
		self.ploc = location
		self.plot_corner()
	
	def plot_corner(self):
		fn = self.get_smd_filename()
		f = b.smd.load(fn)
		g = f[self.ploc]
		if list(g.attrs.keys()).count('description') > 0:
			if g.attrs['description'] == "BIASD MCMC result":
				r = b.smd.read.mcmc(g)
				f.close()
				s = r.chain
				maxauto = np.max((1,int(r.acor.max())))
				s = s[:,::maxauto]
				s = s.reshape((s.shape[0]*s.shape[1],5))
				self.new_corner(s)
				return
				
			elif g.attrs['description'] == 'BIASD Laplace posterior':
				r = b.smd.read.laplace_posterior(g)
				s = np.random.multivariate_normal(r.mu,r.covar,size=10000)
				self.new_corner(s)
				return
		print('This is not a posterior...')
	
	def new_corner(self,s):
		try:
			self.fig.close()
		except:
			pass
		self.fig = FigureCanvas(b.mcmc.plot_corner(s))
		self.fig.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		self.vbox.addWidget(self.fig)
		
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
		elif event.key() == Qt.Key_C:
			self.load_posterior()
		elif event.key() == Qt.Key_S:
			self.savefig()
				

class ui_posterior(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = posterior(self)
		self.setCentralWidget(self.ui)
		self.show()
	
	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()
		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_posterior()
	sys.exit(app.exec_())
