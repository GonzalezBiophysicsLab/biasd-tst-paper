# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to perform MCMC
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

class mcmc(QWidget):
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

		self.sb_burn = QSpinBox()
		self.sb_burn.setRange(0,1000000000)
		self.sb_burn.setValue(100)

		self.sb_steps = QSpinBox()
		self.sb_steps.setRange(1,1000000000)
		self.sb_steps.setValue(1000)

		self.sb_nwalkers = QSpinBox()
		self.sb_nwalkers.setRange(10,1000000)
		self.sb_nwalkers.setValue(30)
		
		form.addRow("Number of Walkers",self.sb_nwalkers)
		form.addRow("Number of Burn-in Steps",self.sb_burn)
		form.addRow("Number of Production Steps",self.sb_steps)
		
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
		self.setWindowTitle('Ensemble Markov chain Monte Carlo')
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
			nburn = self.sb_burn.value()
			nsteps = self.sb_steps.value()
			nwalkers = self.sb_nwalkers.value()
			nthreads = self.parent().parent().prefs.n_threads
			
			if prompt:
				reply = QMessageBox.question(self,'Run?',"Are you sure you want to run this?")
			else:
				reply = QMessageBox.Yes
			if reply == QMessageBox.Yes:
				# try:
				if 1:
					self.parent().parent().parent().statusBar().showMessage("Running %d/%d...."%(self.currenttraj,self.totaltraj))
					
					fn = self.get_smd_filename()
					f = b.smd.load(fn)
					g = f[self.tloc]
					gname = 'MCMC Analysis '+time.ctime()
					gg = g.create_group(gname)
					b.smd.add.parameter_collection(gg,priors,label='Priors')
					gg.attrs['completed'] = 'False'
					gg.attrs['program'] = 'BIASD GUI'
					gg.attrs['number of walkers'] = nwalkers
					gg.attrs['number of burn-in steps'] = nburn
					gg.attrs['number of production steps'] = nsteps
					gg.attrs['number of threads'] = nthreads
					traj = g['data/'+self.dname].value
					traj = traj[np.isfinite(traj)]
					b.smd.save(f)
					
					sampler, ip = b.mcmc.setup(traj, priors, self.tau, nwalkers, initialize='rvs', threads=nthreads)
					
					t0 = time.time()
					if nburn > 0:
						self.parent().parent().parent().statusBar().showMessage("MCMC Burning in %d/%d...."%(self.currenttraj,self.totaltraj))
						sampler, burned = b.mcmc.burn_in(sampler,ip,nsteps=nburn,timer=False)
					else:
						burned = ip

					t1 = time.time()
					self.parent().parent().parent().statusBar().showMessage("MCMC Production %d/%d...."%(self.currenttraj,self.totaltraj))
					sampler = b.mcmc.run(sampler,burned,nsteps=nsteps,timer=False)
					t2 = time.time()

					f = b.smd.load(fn)
					g = f[self.tloc+'/'+gname]
					b.smd.add.mcmc(g,b.mcmc.mcmc_result(sampler), label='Posterior')
					g.attrs['time burn in'] = t1-t0
					g.attrs['time production'] = t2-t1
					g.attrs['completed'] = 'True'
					b.smd.save(f)
					
				# except:
				# 	try:
				# 		f.close()
				# 	except:
				# 		pass
					self.parent().parent().parent().statusBar().showMessage("MCMC Crashed...")
					self.parent().parent().log.new('Crashed MCMC on %s/data/%s'%(self.tloc,self.dname))
				self.parent().parent().parent().statusBar().showMessage("Completed MCMC on %s"%(self.tloc))
				self.parent().parent().log.new('Completed MCMC on %s/data/%s'%(self.tloc,self.dname))
				
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
			self.parent().parent().parent().statusBar().showMessage("Completed batch mode MCMC...")
			QMessageBox.information(self,"Complete","Batch mode complete in %f seconds"%((t1-t0)))

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
		if event.key() == Qt.Key_S:
			self.load_trajectory()
	

class ui_mcmc(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = mcmc(self)
		self.setCentralWidget(self.ui)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_mcmc()
	sys.exit(app.exec_())
