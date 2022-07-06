import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

class CustomWidget(pg.GraphicsWindow):
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    ptr1 = 0
    def __init__(self, parent=None, **kargs):
        pg.GraphicsWindow.__init__(self, **kargs)
        self.setParent(parent)
        self.setWindowTitle('pyqtgraph example: Scrolling Plots')
        p1 = self.addPlot(labels =  {'left':'Voltage', 'bottom':'Time'})
        self.data1 = np.random.normal(size=10)
        self.data2 = np.random.normal(size=10)
        self.curve1 = p1.plot(self.data1, pen=(3,3))
        self.curve2 = p1.plot(self.data2, pen=(2,3))

        timer = pg.QtCore.QTimer(self)
        timer.timeout.connect(self.update)
        timer.start(2000) # number of seconds (every 1000) for next update

    def update(self):
        self.data1[:-1] = self.data1[1:]  # shift data in the array one sample left
                            # (see also: np.roll)
        self.data1[-1] = np.random.normal()
        self.ptr1 += 1
        self.curve1.setData(self.data1)
        self.curve1.setPos(self.ptr1, 0)
        self.data2[:-1] = self.data2[1:]  # shift data in the array one sample left
                            # (see also: np.roll)
        self.data2[-1] = np.random.normal()
        self.curve2.setData(self.data2)
        self.curve2.setPos(self.ptr1,0)

if __name__ == '__main__':
    w = CustomWidget()
    w.show()
    QtGui.QApplication.instance().exec_()