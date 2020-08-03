# -*- coding: utf-8 -*-®
'''
PyQt trace plotter widget
'''
from PyQt5.QtWidgets import QWidget,QSizePolicy

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt

class trace_plotter(FigureCanvas):
	def __init__(self,parent=None):
		self.f, self.a = plt.subplots(1,figsize=(8,4))
		FigureCanvas.__init__(self,self.f)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)

		FigureCanvas.updateGeometry(self)
		self.prep_axis()
		plt.close()

	def prep_axis(self):
		self.a.spines['right'].set_visible(False)
		self.a.spines['top'].set_visible(False)
		self.a.yaxis.set_ticks_position('left')
		self.a.xaxis.set_ticks_position('bottom')

		self.a.set_xlabel('Time (s)')
		self.a.set_ylabel('Signal')

		self.line = self.a.plot([],[])[0]
		self.plot_trace([0,1.],[0,1.])
		self.a.set_title('-/0')
		self.f.tight_layout()
		self.line.set_xdata([])
		self.line.set_ydata([])

		self.draw()

	def plot_trace(self,t,d,title=''):
		if np.ndim(d) == 1:
			if t is None:
				t = np.arange(d.size)
			self.line.set_xdata(t)
			self.line.set_ydata(d)
			self.a.set_xlim(np.min(t),np.max(t))
			deltad = 0.1*(np.max(d)-np.min(d))
			self.a.set_ylim(np.min(d) - deltad, np.max(d) + deltad)
			self.line.set_color('black')
			self.line.set_linewidth(1.)
			self.a.set_title(title)
			self.draw()

	def clear_plot(self):
		self.line.set_xdata([])
		self.line.set_ydata([])
		self.a.set_title('')
		self.a.set_xlabel('')
		self.draw()

	def plot_dist(self,dist_index,pc):
		dnames = [r'$\varepsilon_1$', r'$\varepsilon_2$', r'$\sigma$', r'$k_1$', r'$k_2$']
		colors = ['purple', 'yellow', 'green', 'cyan', 'orange']
		xlabels = ['Signal', 'Signal', 'Signal Noise', r'Rate Constant (s$^{-1}$)', r'Rate Constant (s$^{-1}$)']

		dist_dict = dict(list(zip(list(range(5)),[pc.e1,pc.e2,pc.sigma,pc.k1,pc.k2])))
		dist = dist_dict[dist_index]
		if not dist.okay:
			self.clear_plot()
		else:
			self.a.cla()
			distx = dist.get_ranged_x(1001)
			disty = dist.pdf(distx)

			self.line = self.a.plot(distx,disty,color='k',lw=2)
			# self.line.set_color(self.colors[self.selected])
			self.filledin = self.a.fill_between(distx, disty, color=colors[dist_index], alpha=0.75)

			if dist.name == 'beta':
				self.a.set_xlim(0,1)
			elif dist.name == 'gamma':
				self.a.set_xlim(0,distx[-1])
			else:
				self.a.set_xlim(distx[0],distx[-1])
			if dist.name != 'empty':
				self.a.set_ylim(0.,disty[np.isfinite(disty)].max()*1.2)
			self.a.set_ylabel('Probability',fontsize=18)
			self.a.set_xlabel(xlabels[dist_index],fontsize=18)
			self.a.set_title(dist.label_parameters[0]+": "+str(dist.parameters[0])+", "+dist.label_parameters[1]+": "+str(dist.parameters[1])+r", $E[x] = $"+str(dist.mean()))
			self.draw()
