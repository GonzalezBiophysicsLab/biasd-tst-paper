'''
GUI written in QT5 to explore HDF5 SMD Files
'''
import sys
from h5py import File
from PyQt5.QtWidgets import QApplication, QColumnView, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMainWindow,QMessageBox,QSizePolicy
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QItemSelection

from .plotter import trace_plotter

class selectChangeSignal(QObject):
	signal = pyqtSignal(QItemSelection,QItemSelection)
	
	def sc(self,s,d):
		self.signal.emit(s,d)

class smd_load(QWidget):
	def __init__(self,parent=None,select = True,filename=None):
		super(QWidget,self).__init__(parent)
		if filename is None:
			self.filename = self.parent().parent().get_smd_filename()
		else:
			self.filename = filename
		self.select_type = select
		self.initialize(self.filename,select)
	
	def initialize(self,filename,select):
		self.setWindowTitle('HDF5 SMD Viewer')
		self.setMinimumSize(800,0)
		
		self.viewer = QColumnView()#QTreeView()#QListView()
		self.viewer.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
		
		self.fig = trace_plotter(self)
		self.fig.a.set_title('')
		self.fig.setVisible(False)
		self.fig.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

		if self.select_type:
			b = QPushButton("Select")
			b.clicked.connect(self.select_this)
		bget = QPushButton("Information")
		bget.clicked.connect(self.print_path)
		hbox = QHBoxLayout()

		hbox.addWidget(bget)
		if select:
			hbox.addWidget(b)

		hbox.addStretch(1)
		
		vbox = QVBoxLayout()
		vbox.addWidget(self.viewer)
		
		vbox.addLayout(hbox)
		vbox.addWidget(self.fig)
		vbox.addStretch(1)
		self.setLayout(vbox)
		
		self.scs = selectChangeSignal()
		self.viewer.selectionChanged = self.scs.sc
		self.scs.signal.connect(self.select_change)
		
		self.new_model(filename)
		if not self.model is None:
			self.viewer.setModel(self.model)
			self.show()
			self.adjustSize()
			self.parent().adjustSize()
	
	def select_change(self,selected,deselected):
		try:
			index = selected.indexes()[0]
			ii = self.model.itemFromIndex(index)
			if not ii.data() is None:
				f = File(self.filename,'r')
				self.selected_dataset = f[ii.data()].value
				f.close()
			else:
				self.selected_dataset = None
		except:
			self.selected_dataset = None
			try:
				f.close()
			except:
				pass
		self.update_figure()
		
	def update_figure(self):
		try:
			d = self.selected_dataset
			if d.ndim == 1:
				self.fig.plot_trace(None,d,'')
				self.fig.a.set_xlabel('Frames')
				self.fig.draw()
			self.fig.setVisible(True)
			self.adjustSize()
			self.parent().adjustSize()
		except:
			self.fig.setVisible(False)
			self.adjustSize()
			self.parent().adjustSize()
	
	def closeEvent(self,event):
		self.parent().parent().raise_()
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
			if self.select_type:
				self.select_this()
		elif event.key() == Qt.Key_Escape:
			self.parent().close()
			
	def print_path(self):

		o1 = self.viewer.selectedIndexes()[0].data()
		o2 = self.get_selection(print_it=False)
		if o2 is None:
			o2 = "None"
		else:
			o2 = "['" + o2 + "']"
		QMessageBox.information(self,"Information","Path:\n%s\n\nData:\n%s"%(o2,o1))

	def get_current_path(self):
		try:
			s = self.viewer.selectedIndexes()
			path = []
			if len(s) > 0:
				p = s[0]
				while True:
					if p.data() is None:
						break
					else:
						path.insert(0,p.data())
						p = p.parent()
			child = s[0].child(0,0).data()
			return path,child
		except:
			return [],None
	
	def get_selection(self,print_it=False):
		path,child = self.get_current_path()
		
		if len(path) > 1:
			if child is None:
				child = path[-1]
				path = path[:-1]
			if path.count('attrs') > 0:
				path = path[:path.index('attrs')]
				child = ""
			
			location = ''.join([pp+'/' for pp in path[1:]])[:-1]
			
			if print_it:
				print("'" + location + "'")
			else:
				return location
	
	def select_this(self):
		location = self.get_selection()
		if not location is None:
			self.parent().parent().select_callback(location)
	
	def new_model(self,filename):
		def add_group(group,si):
			attrs = list(group.attrs.items())
			attr_si = QStandardItem('attrs')
			attr_si.setEditable(False)
			for i in range(len(attrs)):
				attr_child = QStandardItem(attrs[i][0])
				attr_child.setEditable(False)
				val = QStandardItem(str(attrs[i][1]))
				val.setEditable(False)
				attr_child.setChild(0,val)
				attr_si.setChild(i,attr_child)
			si.setChild(0,attr_si)
	
			groups  = list(group.items())
			# Add 1 b/c attributes is 0
			for i in range(len(groups)):
				gname = groups[i][0]
				## Add leading zeros for nicer sorting...
				#if gname.startswith('trajectory'):
				#	gname = "trajectory %06d"%(int(gname.split(' ')[-1]))
				newthing = QStandardItem(gname)
				newthing.setEditable(False)
				try:
					add_group(groups[i][1],newthing)
					si.setChild(i+1,newthing)
				except:
					try:
						val = QStandardItem("Dataset - shape: "+str(groups[i][1].value.shape))
						val.setData(groups[i][1].ref)
						val.setEditable(False)
						newthing.setChild(0,val)
						si.setChild(i+1,newthing)
					except:
						print(groups[i][0])
			
		self.model = QStandardItemModel(self.viewer)
		try:
			f = File(filename,'r')
			dataset = QStandardItem(f.filename)
			dataset.setEditable(False)
			add_group(f,dataset)
			f.close()
			self.model.appendRow(dataset)
			
		except:
			pass
			
class ui_loader(QMainWindow):
	def __init__(self,parent=None,select = True,filename=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = smd_load(self,select=select,filename=filename)
		self.setCentralWidget(self.ui)
		self.show()
	
	def closeEvent(self,event):
		try:
			self.parent().activateWindow()
			self.parent().raise_()
			self.parent().setFocus()
		except:
			pass

def launch(fn):
	app = QApplication([])
	w = ui_loader(select=False,filename=fn)
	sys.exit(app.exec_())


if __name__ == '__main__':
	import sys
	launch(sys.argv[1])
