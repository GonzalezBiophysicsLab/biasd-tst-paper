# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to setup preferences
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow, QRadioButton, QGroupBox, QGridLayout, QSpinBox, QFileDialog, QFrame
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QDoubleValidator
from PyQt5.QtCore import Qt, QThread

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import sys

import numpy as np
import biasd as b

class prefs(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()

	def initialize(self):
		layout = QVBoxLayout()

		# Change Likelihood
		vbox = QVBoxLayout()
		likelihoods = ["Python","C","CUDA"]
		self.rbs = [QRadioButton(rtext) for rtext in likelihoods]
		self.change_ll(True)
		[vbox.addWidget(r) for r in self.rbs]
		frame1 = QGroupBox("Likelihood Function")
		frame1.setLayout(vbox)
		layout.addWidget(frame1)

		# Speed Test
		grid1 = QGridLayout()
		self.spin = [QSpinBox(), QSpinBox()]
		[s.setRange(1,1000000000) for s in self.spin]
		self.btest = QPushButton("Test")
		lrepeats = QLabel("Repeats")
		ldatapoints = QLabel("Datapoints")
		self.lavg = QLabel("")
		grid1.addWidget(lrepeats,0,0)
		grid1.addWidget(ldatapoints,0,1)
		grid1.addWidget(self.lavg,0,2)
		grid1.addWidget(self.spin[0],1,0)
		grid1.addWidget(self.spin[1],1,1)
		grid1.addWidget(self.btest,1,2)
		frame2 = QGroupBox("Speed Test Likelihood Function")
		frame2.setLayout(grid1)
		layout.addWidget(frame2)

		# Options
		frame_options = QGroupBox('Options')
		grid2 = QGridLayout()
		leps = QLabel("Numerical Integration Error, ε")
		self.le_eps = QLineEdit()
		self.le_eps.setValidator(QDoubleValidator(1e-300,1e300,100))
		lthreads = QLabel('Number of MCMC Threads')
		self.spin_threads = QSpinBox()
		self.spin_threads.setRange(1,1000000)
		grid2.addWidget(leps,0,0)
		grid2.addWidget(self.le_eps,0,1)
		grid2.addWidget(lthreads,1,0)
		grid2.addWidget(self.spin_threads,1,1)
		frame_options.setLayout(grid2)
		layout.addWidget(frame_options)

		# Reset and Log
		frame3 = QFrame()
		hbox = QHBoxLayout()
		breset = QPushButton("Reset")
		bdumplog = QPushButton("Save Log")
		hbox.addWidget(bdumplog)
		hbox.addWidget(breset)
		frame3.setLayout(hbox)
		layout.addWidget(frame3)

		layout.addStretch(1)
		self.setLayout(layout)

		#Fill Forms
		self.init_forms()

		# Connect Forms & Buttons
		[r.toggled.connect(self.change_ll) for r in self.rbs]
		[s.valueChanged.connect(self.update_speed) for s in self.spin]
		self.btest.clicked.connect(self.test_likelihood)
		#self.le_eps.returnPressed.connect(self.update_eps)
		self.le_eps.editingFinished.connect(self.update_eps)
		self.spin_threads.valueChanged.connect(self.update_threads)
		breset.clicked.connect(self.check_reset)
		bdumplog.clicked.connect(self.save_log)

		self.setWindowTitle('Set Preferences')
		# self.setGeometry(200,200,500,300)
		self.show()

	def update_speed(self):
		p = self.parent().parent().prefs
		p.speed_n = self.spin[0].value()
		p.speed_d = self.spin[1].value()

	def update_threads(self):
		p = self.parent().parent().prefs
		p.n_threads = self.spin_threads.value()
		self.parent().parent().log.new('Updated N threads = '+str(p.n_threads))

	def update_eps(self):
		p = self.parent().parent().prefs
		p.eps = np.array(float(self.le_eps.text()),dtype='float64')
		b.likelihood._eps = p.eps
		self.parent().parent().log.new('Updated epsilon = '+str(p.eps))


	def check_reset(self):
		really = QMessageBox.question(self,"Reset?","Do you really want to reset the preferences?")
		if really == QMessageBox.Yes:
			self.reset()

	def init_forms(self):
		p = self.parent().parent().prefs
		self.spin[0].setValue(p.speed_n)
		self.spin[1].setValue(p.speed_d)
		self.lavg.setText("")
		self.le_eps.setText(str(p.eps))
		self.spin_threads.setValue(p.n_threads)

	def reset(self):
		p = self.parent().parent().prefs
		p.reset()
		self.init_forms()
		self.parent().parent().parent().statusBar().showMessage("Reset Preferences")
		self.parent().parent().log.new("Reset Preferences")


	def save_log(self):
		print(self.parent().parent().log.format())
		oname = QFileDialog.getSaveFileName(self,"Save Log file",'./','*.txt')
		try:
			if not oname[0]:
				return
			f = open(oname[0],'w')
			f.write(self.parent().parent().log.format())
			f.close()
		except:
			QMessageBox.critical(None,"Could Not Save","Could not save file: %s\n."%(oname[0]))

	def speed_tester(self):
		try:
			sb = self.parent().parent().parent().statusBar()
			p = self.parent().parent().prefs
			sb.showMessage('Testing Speed...')
			time = b.likelihood.test_speed(p.speed_n,p.speed_d)
			sout = str(time)+' μsec/datapoint'
			self.lavg.setText(sout)
			self.parent().parent().log.new('Speed Test - '
				+ b.likelihood.ll_version
				+ '\n%d, %d, %s'%(p.speed_n, p.speed_d,sout))
			sb.showMessage('Test Complete')
		except:
			pass

	def test_likelihood(self):
		## Be careful so that you don't lock up people's computers for too long
		if b.likelihood.ll_version == 'Python':
			ev = 1000.
		elif b.likelihood.ll_version == 'C':
			ev = 100.
		elif b.likelihood.ll_version == 'CUDA':
			ev = 10.
		p = self.parent().parent().prefs
		et = ev/1e6 * p.speed_d * p.speed_n
		proceed = True
		if et > 10.:
			really = QMessageBox.question(self,"Long Time",
			"This might take a long time (~ %.0f sec). "%(et)+
			"Are you sure you want to perform this test?")
			if not really == QMessageBox.Yes:
				proceed = False
		if proceed:
			self.speed_tester()

	def change_ll(self,enable):
		try:
			if self.rbs[0].isChecked():
				b.likelihood.use_python_ll()
			elif self.rbs[1].isChecked():
				failure = 'C'
				b.likelihood.use_c_ll()
			elif self.rbs[2].isChecked():
				failure = 'CUDA'
				b.likelihood.use_cuda_ll()
		except:
			QMessageBox.critical(self,"Can't Find %s Library"%(failure),
				"Can't find or load the %s library."%(failure) +
				"Check that it is compiled.")
		for i,t in zip(list(range(3)),['Python','C','CUDA']):
			if b.likelihood.ll_version == t:
				self.rbs[i].setChecked(True)


	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()


class ui_preferences(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = prefs(self)
		self.setCentralWidget(self.ui)
		self.setGeometry(100,100,400,300)
		self.show()

	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()


if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_preferences()
	sys.exit(app.exec_())
