# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to setup independent prior distributions
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow, QSizePolicy, QTabWidget, QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QDoubleValidator
from PyQt5.QtCore import Qt

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import sys

import biasd as b

from .smd_loader import ui_loader
from .plotter import trace_plotter


class previewer(QWidget):
	def __init__(self,parent=None,dists = None):
		super(QWidget,self).__init__(parent)
		self.dists = dists
		self.dist_names = ['ϵ_1','ϵ_2','  σ','k_1','k_2']
		
		if not dists is None:
			self.initialize()
		else:
			self.parent().close()
	
	def initialize(self):
		vbox = QVBoxLayout()
		hbox = QHBoxLayout()
		self.cb = QComboBox()
		[self.cb.addItem(i) for i in self.dist_names]
		self.cb.setCurrentIndex(0)
		self.cb.currentIndexChanged.connect(self.update_plot)

		hbox.addWidget(self.cb)
		bsave = QPushButton('Save Figure')
		bsave.clicked.connect(self.savefig)
		hbox.addWidget(bsave)
		hbox.addStretch(1)
		vbox.addLayout(hbox)
		
		self.fig = trace_plotter()
		vbox.addWidget(self.fig)
		self.update_plot()
		
		self.cb.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)
		self.fig.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		
		self.setLayout(vbox)
		self.setFocus()
		self.show()
	
	def savefig(self):
		oname = QFileDialog.getSaveFileName(self,"Save Figure",'./','jpg (*.jpg);;png (*.png);;pdf (*.pdf);;eps (*.eps)')
		self.setFocus()
		try:
			self.fig.f.savefig(oname[0])
		except:
			QMessageBox.critical(None,"Could Not Save","Could not save file: %s\n."%(oname[0]))
	
	def update_plot(self):
		i = self.cb.currentIndex()
		self.fig.plot_dist(i,self.dists)
	
	def jump(self,di):
		i = self.cb.currentIndex() + di
		if i >= self.cb.count():
			i = self.cb.count() - 1
		elif i < 0:
			i = 0
		self.cb.setCurrentIndex(i)
		self.update_plot()
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
		elif event.key() == Qt.Key_Left:
			self.jump(-1)
		elif event.key() == Qt.Key_Right:
			self.jump(1)
		elif event.key() == Qt.Key_S:
			self.savefig()
		

class ui_preview(QMainWindow):
	def __init__(self,parent=None,dists=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = previewer(self,dists)
		self.setCentralWidget(self.ui)
		self.show()
		
	def keyPressEvent(self,event):
		self.ui.keyPressEvent(event)
	
	def closeEvent(self,event):
		self.ui.close()
		self.parent().setFocus()
		self.parent().raise_()
		self.parent().activateWindow()
		self.close()
		


class distribution(QWidget):
	def __init__(self,dist_name):
		super(QWidget,self).__init__()
		self.dist_types = ['beta','empty','gamma','normal','uniform']
		self.dist_fxns = [b.distributions.beta,
			b.distributions.empty,
			b.distributions.gamma,
			b.distributions.normal,
			b.distributions.uniform
		]
		self.initialize(dist_name)

	def initialize(self,dist_name):
		self.hbox = QHBoxLayout()
		
		distlabel = QLabel(dist_name)
		
		self.prior_type = QComboBox()
		[self.prior_type.addItem(i) for i in self.dist_types]
#		self.prior_type.activated.connect(self.setFocus)
		
		self.param1 = QLineEdit()
		self.param2 = QLineEdit()
		[p.setValidator(QDoubleValidator(-1e300,1e300,100)) for p in [self.param1,self.param2]]
		[p.returnPressed.connect(self.setFocus) for p in [self.param1,self.param2]]
		
		self.hbox.addStretch(1)
		self.hbox.addWidget(distlabel)
		self.hbox.addWidget(self.prior_type)
		self.hbox.addWidget(self.param1)
		self.hbox.addWidget(self.param2)
		
		self.setLayout(self.hbox)
		self.show()
	
	def get_distribution(self):
		dtype = str(self.prior_type.currentText())
		if dtype == 'empty':
			pp1 = 0.
			pp2 = 0.
		else:
			pp1 = float(self.param1.text())
			pp2 = float(self.param2.text())
		fxn = self.dist_fxns[self.dist_types.index(dtype)]
		return fxn(pp1,pp2)
	
class priors(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.priors_dists = ['ϵ_1','ϵ_2','  σ','k_1','k_2']
		self.initialize()

	def initialize(self):
		self.setWindowTitle('Set Priors')

		qw = QWidget()
		vbox = QVBoxLayout()
		hbox = QHBoxLayout()
		
		bload = QPushButton("Load")
		bcheck = QPushButton("Check")
		bset = QPushButton("Set")
		bview = QPushButton("View")

		bload.clicked.connect(self.load_prior)
		bcheck.clicked.connect(lambda:self.check_dist(warn=True))
		bset.clicked.connect(self.set_dist)
		bview.clicked.connect(self.view_dist)

		hbox.addStretch(1)
		[hbox.addWidget(i) for i in [bload,bcheck,bset]]
		hbox.addWidget(bview)
		
		vbox.addLayout(hbox)
		
		self.dists = [distribution(self.priors_dists[i]) for i in range(5)]
		[vbox.addWidget(self.dists[i]) for i in range(5)]
		self.update_dists()
		
		vbox.addStretch(1)
		self.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Expanding)
		# qw.adjustSize()
		# qw.setLayout(vbox)
		self.setLayout(vbox)

		# hbox2 = QHBoxLayout()
		# hbox2.addWidget(qw)
		# self.fig = trace_plotter(self)
		# self.fig.clear_plot()
		# self.fig.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		# self.fig.setVisible(False)
		# hbox2.addWidget(self.fig)
		# hbox2.addStretch(1)
		# self.setLayout(hbox2)

		self.parent().adjustSize()
		self.show()
		
	def update_dists(self):
		try:
			p = self.parent().parent().priors
			for i,pp in zip(list(range(5)),[p.e1,p.e2,p.sigma,p.k1,p.k2]):
				cb = self.dists[i].prior_type
				cb.setCurrentIndex(cb.findText(pp.name))
				self.dists[i].param1.setText(str(pp.parameters[0]))
				self.dists[i].param2.setText(str(pp.parameters[1]))
		except:
			QMessageBox.critical(self,"Priors are misformed","Priors are misformed. Cannot load them. Update them here.")

	def get_smd_filename(self):
		try:
			return self.parent().parent().filename
		except:
			return None
	
	def load_prior(self):
		try:
			self.loader.close()
		except:
			pass
		
		self.loader = ui_loader(self)
		self.loader.show()
	
	def select_callback(self,loc):
		self.loader.close()
		try:
			import h5py
			fn = self.get_smd_filename()
			f = h5py.File(fn,'r')
			g = f[loc]
			if g.attrs['description'] == 'BIASD parameter collection':
				p = b.smd.read.parameter_collection(g)
				self.parent().parent().priors = p
				self.update_dists()
				n,p = p.format()
				logstring = "Loaded Priors from "+fn+"[" + loc+ "]" + ''.join(["\n "+nn+" - "+ str(pp) for nn,pp in zip(n,p)])
				self.parent().parent().log.new(logstring)
				self.parent().parent().parent().statusBar().showMessage("Priors Loaded")
			else:
				QMessageBox.critical(self,"Not a parameter collection","This doesn't appear to be a parameter collection")
				f.close()

		except:
			try:
				f.close()
			except:
				pass
			QMessageBox.critical(self,"Couldn't load this collection","Couldn't load this from the HDF5 file as a parameter collection")
	
	def get_dists(self):
		try:
			bd = []
			dist_check = 0
			for i in range(5):
				bd.append(self.dists[i].get_distribution())
			dist_check = 1
			pc = b.distributions.parameter_collection(*bd)
			pc.check_dists()
			return pc
		except:
			if dist_check:
				QMessageBox.critical(self,
					"Out of Bounds",
					"Something is out of bounds in %s. Check support of those distribution parameters."%(''.join([self.priors_dists[i] +" " for i in range(5) if not bd[i].okay]))
				)
			else:
				QMessageBox.critical(self,
					"Something is Malformed ",
					"Something is Malformed in %s"%(self.priors_dists[i])
				)
	
	def check_dist(self,warn=True):
		params = self.get_dists()
		if not params is None:
			if params.okay:
				if warn:
					QMessageBox.information(self,
						"It's okay",
						"These Distributions are okay."
					)
				return params
		return None
		
	def view_dist(self):
		d = self.check_dist(warn=False)
		if not d is None:
			try:
				self.ui_preview.close()
			except:
				pass
			self.ui_preview = ui_preview(self,d)
			self.ui_preview.show()
				
	def set_dist(self):
		params = self.check_dist(warn=False)
		if not params is None:
			try:
				self.parent().parent().priors = params
				self.update_dists()
				self.parent().parent().parent().statusBar().showMessage("Priors Set")
				n,p = params.format()
				logstring = "Set Priors" + ''.join(["\n "+nn+" - "+ str(pp) for nn,pp in zip(n,p)])
				self.parent().parent().log.new(logstring)
			except:
				print('set nothing')
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
		elif event.key() == Qt.Key_L:
			self.load_prior()
		elif event.key() == Qt.Key_V:
			self.view_dist()
		elif event.key() == Qt.Key_C:
			self.check_dist()
		elif event.key() == Qt.Key_S:
			self.set_dist()
			

class ui_priors(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = priors(self)
		self.setCentralWidget(self.ui)
		self.show()
	
	def closeEvent(self,event):
		try:
			self.ui.loader.close()
		except:
			pass

		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_priors()
	sys.exit(app.exec_())
