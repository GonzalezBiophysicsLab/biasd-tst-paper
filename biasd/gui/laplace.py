# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to perform the Laplace Approximation
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow, QShortcut, QGroupBox,QFormLayout,QSpinBox
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
import time
# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import sys

import biasd as b

from .smd_loader import ui_loader

class laplace(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()

	def initialize(self):
	
		vbox = QVBoxLayout()
		
		bload = QPushButton("Select Trajectory")
		self.brun = QPushButton("Run single")
		self.bbatch = QPushButton("Batch all like this")
		gb = QGroupBox('Options')
		form = QFormLayout()
		self.sb_nrestarts = QSpinBox()
		self.sb_nrestarts.setRange(1,100)
		self.sb_nrestarts.setValue(5)
		form.addRow("Number of Restarts",self.sb_nrestarts)
		gb.setLayout(form)
		
		bload.clicked.connect(self.load_trajectory)
		self.brun.clicked.connect(self.run)
		self.bbatch.clicked.connect(self.batch)
		[b.setEnabled(False) for b in [self.brun,self.bbatch]]
		vbox.addWidget(bload)
		vbox.addWidget(gb)
		hbox = QHBoxLayout()
		hbox.addStretch(1)
		hbox.addWidget(self.brun)
		hbox.addWidget(self.bbatch)
		hbox.addStretch(1)
		vbox.addLayout(hbox)
		vbox.addStretch(1)
		self.setLayout(vbox)
		self.setFocus()
		self.setWindowTitle('Laplace Approximation')
		self.show()
	
	def get_smd_filename(self):
		return self.parent().parent().get_smd_filename()
	
	def load_trajectory(self):
		try:
			self.loader.close()
		except:
			pass
		self.loader = ui_loader(self,select=True)
		self.loader.show()
		
	def select_callback(self,location):
		self.loader.close()
		l = location.split('/')
		if len(l) == 3:
			if l[0].startswith('trajectory') and l[1] == 'data':
				self.tloc = l[0]
				self.dname = l[2]
				self.get_tau()
				self.parent().parent().parent().statusBar().showMessage("Chose %s"%(self.tloc))
				[b.setEnabled(True) for b in [self.brun,self.bbatch]]
				self.totaltraj = 1
				self.currenttraj = 1
				return
		else:
			QMessageBox.critical(self,"Couldn't load this dataset","Couldn't load this from the HDF5 file as a dataset. Make sure you are selecting a dataset.")

	def get_tau(self):
		try:
			fn = self.get_smd_filename()
			f = b.smd.load(fn)
			if list(f[self.tloc+'/data'].keys()).count('Time') > 0:
				t = f[self.tloc+'/data/Time'].value
			elif list(f[self.tloc+'/data'].keys()).count('time') > 0:
				t = f[self.tloc+'/data/time'].value
			self.tau = t[1] - t[0]
			f.close()
		except:
			try:
				f.close()
			except:
				pass
			self.tau = None

	def run(self,prompt=True):
		if not self.tloc is None:
			priors = self.parent().parent().priors
			nr = self.sb_nrestarts.value()
			if prompt:
				reply = QMessageBox.question(self,'Run?',"Are you sure you want to run this?")
			else:
				reply = QMessageBox.Yes
			if reply == QMessageBox.Yes:
				try:
					self.parent().parent().parent().statusBar().showMessage("Running %d/%d...."%(self.currenttraj,self.totaltraj))
					fn = self.get_smd_filename()
					f = b.smd.load(fn)
					g = f[self.tloc]
					gname = 'Laplace Analysis '+time.ctime()
					gg = g.create_group(gname)
					b.smd.add.parameter_collection(gg,priors,label='Priors')
					gg.attrs['completed'] = 'False'
					gg.attrs['program'] = 'BIASD GUI'
					traj = g['data/'+self.dname].value
					traj = traj[np.isfinite(traj)]
					b.smd.save(f)
					result = b.laplace.laplace_approximation(traj, priors, self.tau, nrestarts=nr)
					f = b.smd.load(fn)
					g = f[self.tloc+'/'+gname]
					b.smd.add.laplace_posterior(g,result,label='Posterior')
					g.attrs['completed'] = 'True'
					g.attrs['number of restarts'] = nr
					b.smd.save(f)
				except:
					try:
						f.close()
					except:
						pass
					self.parent().parent().parent().statusBar().showMessage("Laplace Crashed...")
					self.parent().parent().log.new('Crashed Laplace Approximation on %s/data/%s'%(self.tloc,self.dname))
				self.parent().parent().parent().statusBar().showMessage("Completed Laplace on %s"%(self.tloc))
				self.parent().parent().log.new('Completed Laplace Approximation on %s/data/%s'%(self.tloc,self.dname))
				
	def batch(self):
		reply = QMessageBox.question(self,'Run?',"Are you sure you want to process all of the trajectories like this?")
		if reply == QMessageBox.Yes:
			fn = self.get_smd_filename()
			f = b.smd.load(fn)
			tlist = [i for i in list(f.keys()) if i.startswith('trajectory')]
			f.close()
			self.totaltraj = len(tlist)
			t0 = time.time()
			for t in tlist:
				try:
					self.tloc = t
					self.get_tau()
					self.run(prompt=False)
					self.currenttraj += 1
				except:
					pass
			t1 = time.time()
			self.parent().parent().parent().statusBar().showMessage("Completed batch mode Laplace...")
			QMessageBox.information(self,"Complete","Batch mode complete in %f seconds"%((t1-t0)))

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
		if event.key() == Qt.Key_S:
			self.load_trajectory()
	

class ui_laplace(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = laplace(self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_laplace()
	sys.exit(app.exec_())
