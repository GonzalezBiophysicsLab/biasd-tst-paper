# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to setup preferences
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow,QSizePolicy, QFileDialog, QDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QDoubleValidator
from PyQt5.QtCore import Qt, QTimer

import numpy as np

import sys
#biasd_path = '/Users/colin/Desktop/20161220 biasd_release/biasd'

import biasd as b

from .smd_loader import smd_load
from .plotter import trace_plotter

class ui_set_tau(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent=parent)
		self.initialize()

	def initialize(self):
		p = self.parent().parent().parent().prefs
		tau = p.tau
		
		self.statusBar()
		
		qw = QWidget()
		hbox = QHBoxLayout()
		
		ltau = QLabel("τ (s):")
		
		self.le_tau = QLineEdit()
		self.le_tau.setValidator(QDoubleValidator(1e-300,1e300,100))
		self.le_tau.editingFinished.connect(self.update_tau)
		self.le_tau.setText(str(tau))

		bset = QPushButton('Set')
		bset.clicked.connect(self.update_tau)
		
		[hbox.addWidget(bb) for bb in [ltau,self.le_tau,bset]]
		qw.setLayout(hbox)
		self.setCentralWidget(qw)
		self.setWindowTitle('Set tau')
		self.show()
		
		self.le_tau.selectAll()
	
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.close()
	
	def closeEvent(self,event):
		# self.ui.close()
		self.parent().setFocus()
		self.parent().raise_()
		self.parent().activateWindow()
		self.close()
	
	def update_tau(self):
		try:
			p = self.parent().parent().parent().prefs
			new_tau = float(self.le_tau.text())
			if p.tau != new_tau:
				p.tau = new_tau
				self.parent().parent().parent().log.new('Updated tau = '+str(p.tau))
				self.close()
			else:
				self.statusBar().showMessage("There's no change...")
			self.parent().update()
		except:
			pass
		self.setFocus()

	
class traces(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()

	def initialize(self):
		
		# Buttons | Plot
		hbox = QHBoxLayout()
		# Buttons
		qwbuttons = QWidget()
		vbox = QVBoxLayout()
		
		badd = QPushButton('Import')
		btranspose = QPushButton('Transpose')
		bprev = QPushButton('<<')
		bnext = QPushButton('>>')
		bappend = QPushButton('Commit')
		
		bprev.clicked.connect(self.prev_trace)
		bnext.clicked.connect(self.next_trace)
		btranspose.clicked.connect(self.transpose_traces)
		bappend.clicked.connect(self.append_to_smd)
		badd.clicked.connect(self.add_traces)

		qwtemp = QWidget()
		hbox3 = QHBoxLayout()
		[hbox3.addWidget(bb) for bb in [badd,btranspose]]
		hbox3.addStretch(1)
		qwtemp.setLayout(hbox3)
		vbox.addWidget(qwtemp)

		qwtemp = QWidget()
		hbox2 = QHBoxLayout()
		hbox2.addStretch(1)
		[hbox2.addWidget(bb) for bb in [bprev,bnext]]
		hbox2.addStretch(1)
		qwtemp.setLayout(hbox2)
		vbox.addWidget(qwtemp)
		
		bset_tau = QPushButton('&Set τ')
		bset_tau.clicked.connect(self.launch_set_tau)
		vbox.addWidget(bset_tau)
		
		vbox.addStretch(1)
		
		vbox.addWidget(bappend)
		
		qwbuttons.setLayout(vbox)
		hbox.addWidget(qwbuttons)
		
		# Plot
		self.fig = trace_plotter(self)
		hbox.addWidget(self.fig)
		self.stretcher = hbox.addStretch(1)
		
		self.buttons = [badd,btranspose,bprev,bnext,bappend,bset_tau]
		
		self.setLayout(hbox)
		self.setWindowTitle('Import Traces')
		self.setFocus()
		self.show()
	
	def launch_set_tau(self):
		try:
			self.ui_tau.close()
		except:
			pass
		self.ui_tau = ui_set_tau(self)
	
	def transpose_traces(self):
		try:
			d1,d2 = self.imported_data.shape
			if d1 > 1 and d2 > 1:
				self.imported_data = self.imported_data.T
				self.trace_index = 0
				self.update_figure()
		except:
			pass
	
	def jump_trace(self,i):
		try:
			ni = int(i + self.trace_index)
			if ni < 0:
				ni = 0
			elif ni >= self.imported_data.shape[0]:
				ni = self.imported_data.shape[0] -1
			self.trace_index = ni
			self.update_figure()
		except:
			pass
	
	def next_trace(self):
		self.jump_trace(1)
		
	def prev_trace(self):
		self.jump_trace(-1)
		
	def update_figure(self):
		tau = self.get_safe_tau()
		d = self.imported_data[self.trace_index]
		title = str(self.trace_index) + "/"+str(self.imported_data.shape[0]-1)
		self.fig.plot_trace(np.arange(d.size)*tau,d,title=title)
		self.setFocus()
	
	def update(self):
		self.update_figure()
	
	def get_safe_tau(self):
		try:
			p = self.parent().parent().prefs
			tau = p.tau
		except:
			tau = 1.
		return tau
	
	def add_traces(self):
		fname = str(QFileDialog.getOpenFileName(self,
			"Choose data file to load trajectories from","./")[0])
		data = False
		if fname:
			try:
				self.imported_data = b.smd.loadtxt(fname)
				if self.imported_data.ndim == 1:
					self.imported_data = self.imported_data[np.newaxis,:]
				data = True
			except:
				QMessageBox.critical(self,'Could not Load',
					'Could not load file: %s. It should be either an (N,T) or (T,N) array with tab, space, or comma delimiters. It can also be a binary numpy ndarray.'%(fname))
		if data:
			self.trace_index = 0
			self.update_figure()
			
	def append_to_smd(self):
		doit = False
		d = self.imported_data
		fname = self.parent().parent().filename
		try:
			f = b.smd.load(fname)
			# check if this is maybe the same
			doit = True
			if list(f.attrs.keys()).count('number of trajectories'):
				nn = f.attrs['number of trajectories']
				if nn == d.shape[0]:
					really = QMessageBox.question(self,'Are you sure?', 'It looks like %s already contains %d trajectories. Are you sure you want to *APPEND* these new trajectories? You will have %d trajectories then.'%(fname, nn, 2*nn))
					if really != QMessageBox.Yes:
						doit = False
		except:
			try:
				f.close()
			except:
				pass
			
		if doit:
			tau = self.parent().parent().prefs.tau
			t = np.arange(d.shape[1])*tau
			b.smd.add.trajectories(f,t,d)
			for i in range(d.shape[0]):
				group = f['trajectory %d'%(i)]
				group.attrs['tau'] = tau
				group.attrs['added by'] = 'BIASD GUI v.'+self.parent().parent().__version__
			
			#Be REAL safe
			try:
				b.smd.save(f)
				self.parent().parent().log.new('Appended %d Trajectories onto: \n    %s '%(d.shape[0],fname))
				self.parent().parent().parent().statusBar().showMessage("%d New Trajectories Added..."%(d.shape[0]))
			except:
				pass
		
	
	def closeEvent(self,event):
		self.close()
		self.parent().close()
	
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			for b in self.buttons:
				if b.hasFocus():
					self.setFocus()
					return
			self.close()
			self.parent().close()
		elif event.key() == Qt.Key_Right:
			self.next_trace()
		elif event.key() == Qt.Key_Left:
			self.prev_trace()
		elif event.key() == Qt.Key_Up:
			try:
				d = self.imported_data.shape[0]
				self.jump_trace(-d)
			except:
				pass
		elif event.key() == Qt.Key_Down:
			try:
				d = self.imported_data.shape[0]
				self.jump_trace(d)
			except:
				pass
		
		elif event.key() == Qt.Key_I:
			self.add_traces()
		elif event.key() == Qt.Key_C:
			self.append_to_smd()
		elif event.key() == Qt.Key_S:
			self.launch_set_tau()
		elif event.key() == Qt.Key_T:
			self.transpose_traces()

class ui_traces(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = traces(self)
		# self.setGeometry(100,100,600,400)
		self.setCentralWidget(self.ui)
		self.setFocus()
		
		self.show()
	
	def keyPressEvent(self,event):
		self.ui.keyPressEvent(event)
	
	def closeEvent(self,event):
		self.ui.close()
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()
		self.close()
		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_traces()
	sys.exit(app.exec_())
