# -*- coding: utf-8 -*-®
import sys, os

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

# PyQt5 imports
from PyQt5.QtWidgets import QApplication, QColumnView, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMainWindow, QFileDialog, QLabel, QMessageBox, QAction, QShortcut
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QIcon
from PyQt5.QtCore import Qt, QTimer

# BIASD Path
this_path = os.path.dirname(os.path.abspath(__file__))
biasd_path = this_path[:-len("/biasd/gui")]
sys.path.append(biasd_path)
import biasd as b

# Other UIs 
from .priors2 import ui_priors
from .smd_loader import ui_loader
from .preferences import ui_preferences
from .traces import ui_traces, ui_set_tau
from .laplace import ui_laplace
from .mcmc import ui_mcmc
from .posterior import ui_posterior

__version__ = "0.1.1"

class _logfile():
	from time import ctime
	def __init__(self):
		self.log = []
	def new(self,entry):
		try:
			self.log.append(_logfile.ctime() + " - " + str(entry))
		except:
			pass
	def format(self):
		return ''.join(ll+"\n" for ll in self.log)

class prefs():
	def __init__(self):
		class _x(): pass
		self.default = _x()
		self.default.eps = b.likelihood._eps
		self.default.n_threads = 1
		self.default.speed_n = 10
		self.default.speed_d = 5000
		self.default.tau = 1.
		self.reset()

	def reset(self):
		self.eps = self.default.eps
		self.n_threads = self.default.n_threads
		self.speed_n = self.default.speed_n
		self.speed_d = self.default.speed_d
		self.tau = self.default.tau

class biasd_control(QWidget):
	def __init__(self,parent=None):
		super(QWidget,self).__init__(parent)
		self.initialize()
		self.__version__ = __version__

	def initialize(self):

		self.bprior = QPushButton("&Priors")	
		bnew = QPushButton("&New")
		bload = QPushButton("&Open")
		bexplore = QPushButton("&Explore")
		breset = QPushButton("Reset")
		bprefs = QPushButton('Pre&ferences')
		self.btraces = QPushButton('&Traces')
		self.blaplace = QPushButton('&Laplace')
		self.bmcmc = QPushButton("&MCMC")
		self.bposterior = QPushButton("&View Posterior")

		# Overall Layout
		vbox = QVBoxLayout()
		
		# Box 1
		qtemp = QWidget()
		hbox1 = QHBoxLayout()
		[hbox1.addWidget(bbb) for bbb in [bnew,bload,bexplore]]
		hbox1.addStretch(1)
		qtemp.setLayout(hbox1)
		vbox.addWidget(qtemp)
		
		# Middle Buttons
		vbox.addWidget(self.btraces)
		vbox.addWidget(self.bprior)
		
		# box analysis
		qtemp = QWidget()
		hbox3 = QHBoxLayout()
		[hbox3.addWidget(bbb) for bbb in [self.blaplace,self.bmcmc]]
		qtemp.setLayout(hbox3)
		vbox.addWidget(qtemp)
		vbox.addWidget(self.bposterior)
		
		# Box 2
		qtemp = QWidget()
		hbox2 = QHBoxLayout()
		[hbox2.addWidget(bbb) for bbb in [breset,bprefs]]
		qtemp.setLayout(hbox2)
		vbox.addWidget(qtemp)
		
		### TESTING PURPOSES
		#btest = QPushButton('... print log ...')
		#vbox.addWidget(btest)
		#btest.clicked.connect(self.test)
		
		
		self.setLayout(vbox)
		
		## Connect the buttons
		self.bprior.clicked.connect(self.launch_priors)
		bnew.clicked.connect(self.new_smd)
		bload.clicked.connect(self.load_smd)
		bexplore.clicked.connect(self.explore_smd)
		bprefs.clicked.connect(self.launch_preferences)
		breset.clicked.connect(self.reset)
		self.btraces.clicked.connect(self.launch_traces)
		self.blaplace.clicked.connect(self.launch_laplace)
		self.bmcmc.clicked.connect(self.launch_mcmc)
		self.bposterior.clicked.connect(self.launch_posterior)
		
		[bbb.setEnabled(False) for bbb in [self.blaplace,self.bmcmc,self.bposterior,self.bprior,self.btraces]]
		
		self.priors = b.distributions.empty_parameter_collection()
		self.log = _logfile()
		self.prefs = prefs()
		
		self.init_shortcuts()
		
		self.filename = ''
	
	def launch_laplace(self):
		try:
			self.ui_laplace.close()
		except:
			pass
		self.ui_laplace = ui_laplace(self)
		self.ui_laplace.show()
		
	def launch_mcmc(self):
		try:
			self.ui_mcmc.close()
		except:
			pass
		self.ui_mcmc = ui_mcmc(self)
		self.ui_mcmc.show()
		
	def launch_posterior(self):
		try:
			self.ui_posterior.close()
		except:
			pass
		self.ui_posterior = ui_posterior(self)
		self.ui_posterior.show()
		
	def make_shortcut(self,key,fxn):
		qs = QShortcut(self)
		qs.setKey(key)
		qs.activated.connect(fxn)
	
	def quit_wrapper(self):
		reply = QMessageBox.question(self,'Quit',"Are you sure you want to quit?")
		if reply == QMessageBox.Yes:
			self.parent().close()
		
	def init_shortcuts(self):
		# quit
		self.make_shortcut("Ctrl+Q",self.quit_wrapper)
		self.make_shortcut("o",self.load_smd)
		self.make_shortcut("p",self.launch_priors)
		self.make_shortcut("t",self.launch_traces)
		self.make_shortcut("e",self.explore_smd)
		self.make_shortcut("f",self.launch_preferences)
		self.make_shortcut("l",self.launch_laplace)
		self.make_shortcut("m",self.launch_mcmc)
		self.make_shortcut("v",self.launch_posterior)

	
	def launch_traces(self):
		if self.filename != "":
			self.ui_traces = ui_traces(self)
		
	
	def reset(self):
		try:
			self.ui_explore.close()
		except:
			pass
		try:
			self.ui_priors.close()
		except:
			pass
		try:
			self.ui_traces.close()
		except:
			pass
		try:
			self.ui_prefs.close()
		except:
			pass
		self.filename = ""
		self.priors = b.distributions.empty_parameter_collection()
		[bbb.setEnabled(False) for bbb in [self.blaplace,self.bmcmc,self.bposterior,self.bprior,self.btraces]]
		self.parent().statusBar().showMessage('Reset')

		self.log.new('Reset GUI')
	
	def new_smd(self):
		oname = QFileDialog.getSaveFileName(self,"Create new HDF5 SMD file",'./','HDF SMD (*.hdf5 *.SMD *.smd *.HDF5 *.HDF *.hdf *.dat *.biasd)')

		try:
			if not oname[0]:
				return
			f = b.smd.new(oname[0],force=True)
			b.smd.save(f)
			self.reset()
			self.set_filename(oname[0])
			[bbb.setEnabled(True) for bbb in [self.blaplace,self.bmcmc,self.bposterior,self.bprior,self.btraces]]
		except:
			QMessageBox.critical(None,"Could Not Make New File","Could not make new file: %s\n."%(oname[0]))
		
	def load_smd(self):
		fname = str(QFileDialog.getOpenFileName(self,"Choose HDF5 SMD file to load","./",filter='HDF5 SMD (*.smd *.SMD *.HDF5 *.HDF *.hdf5 *.hdf *.dat *.biasd)')[0])
		data = False
		if fname:
			try:
				f = b.smd.load(fname)
				f.close()
				data = True
			except:
				QMessageBox.critical(None,'Could Not Load File','Could not load file: %s.\nMake sure to use an HDF5 file'%fname)
		if data:
			self.reset()
			self.set_filename(fname)
			[bbb.setEnabled(True) for bbb in [self.blaplace,self.bmcmc,self.bposterior,self.bprior,self.btraces]]
			
	def set_filename(self,fname):
		self.filename = fname
		if len(self.filename) > 25:
			dispfname ="....."+self.filename[-25:]
		else:
			dispfname = self.filename
		# self.lfilename.setText(dispfname)
		self.parent().statusBar().showMessage('Loaded %s'%(dispfname))
		self.log.new('Loaded %s'%(self.filename))
	
	def get_smd_filename(self):
		return self.filename

	def explore_smd(self):
		try:
			self.ui_explore.close()
		except:
			pass
		self.ui_explore = ui_loader(parent = self, select = False)
		self.ui_explore.setWindowTitle('Explore')
		self.ui_explore.show()
		
	
	def launch_preferences(self):
		try:
			if not self.ui_prefs.isVisible():
				self.ui_pref.setVisible(True)
			self.ui_prefs.raise_()
		except:
			self.ui_prefs = ui_preferences(self)
			self.ui_prefs.setWindowTitle('Preferences')
			self.ui_prefs.show()
	
	
	def test(self):
		# self.ui_priors.ui.update_dists()
		print(self.log.format())
		# print self.prefs.n_threads,self.prefs.eps,self.prefs.speed_n,self.prefs.speed_d,self.prefs.default.speed_d
	
	def launch_priors(self):
		try:
			if not self.ui_priors.isVisible():
				self.ui_priors.setVisible(True)
			self.ui_priors.raise_()
		except:
			self.ui_priors = ui_priors(self)
			self.ui_priors.setWindowTitle('Priors')
			self.ui_priors.show()
	
class gui(QMainWindow):
	def __init__(self):
		super(QMainWindow,self).__init__()
		self.initialize()

	def initialize(self):
		self.bc = biasd_control(self)
		self.setCentralWidget(self.bc)
		self.statusBar().showMessage("Ready")
		
		global __version__, this_path
		self.setWindowTitle("BIASD - "+__version__)
		self.setWindowIcon(QIcon(this_path+'/icon-01.png'))
		# self.setGeometry(250,250,250,150)
#		self.move(0,0)
		self.show()
	
def launch():
	app = QApplication([])
	app.setStyle('fusion')
	g = gui()
	app.setWindowIcon(g.windowIcon())
	sys.exit(app.exec_())
	
if __name__ == '__main__':
	launch()
