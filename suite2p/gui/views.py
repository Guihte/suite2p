"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QPushButton, QSlider, QButtonGroup, QLabel, QStyle, QStyleOptionSlider, QApplication, QAbstractSlider
from qtpy.QtGui import QPainter

from .. import registration


def _slider_no_ticks():
    # Qt5 exposes this on QSlider; newer bindings may nest it under TickPosition.
    tick_position = getattr(QSlider, "TickPosition", None)
    if tick_position is not None and hasattr(tick_position, "NoTicks"):
        return tick_position.NoTicks
    return QSlider.NoTicks


def _qstyle_enum(name, enum_name):
    if hasattr(QStyle, name):
        return getattr(QStyle, name)
    enum = getattr(QStyle, enum_name, None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    raise AttributeError(f"QStyle does not expose {name}")


def _qabstractslider_enum(name):
    if hasattr(QAbstractSlider, name):
        return getattr(QAbstractSlider, name)
    enum = getattr(QAbstractSlider, "SliderAction", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    raise AttributeError(f"QAbstractSlider does not expose {name}")


CC_SLIDER = _qstyle_enum("CC_Slider", "ComplexControl")
SC_NONE = _qstyle_enum("SC_None", "SubControl")
SC_SLIDER_GROOVE = _qstyle_enum("SC_SliderGroove", "SubControl")
SC_SLIDER_HANDLE = _qstyle_enum("SC_SliderHandle", "SubControl")
SC_SLIDER_TICKMARKS = _qstyle_enum("SC_SliderTickmarks", "SubControl")
STATE_SUNKEN = _qstyle_enum("State_Sunken", "StateFlag")
SLIDER_MOVE = _qabstractslider_enum("SliderMove")
SLIDER_NO_ACTION = _qabstractslider_enum("SliderNoAction")


def make_buttons(parent):
    """ view buttons"""
    # view buttons
    parent.view_names = [
        "Q: ROIs",
        "W: mean img",
        "E: mean img (enhanced)",
        "R: correlation map",
        "T: max projection",
        "Y: mean img chan2, corr",
        "U: mean img chan2",
    ]
    b = 0
    parent.viewbtns = QButtonGroup(parent)
    vlabel = QLabel(parent)
    vlabel.setText("<font color='white'>Background</font>")
    vlabel.setFont(parent.boldfont)
    vlabel.resize(vlabel.minimumSizeHint())
    parent.l0.addWidget(vlabel, 1, 0, 1, 1)
    for names in parent.view_names:
        btn = ViewButton(b, "&" + names, parent)
        parent.viewbtns.addButton(btn, b)
        if b > 0:
            parent.l0.addWidget(btn, b + 2, 0, 1, 1)
        else:
            parent.l0.addWidget(btn, b + 2, 0, 1, 1)
            label = QLabel("sat: ")
            label.setStyleSheet("color: white;")
            parent.l0.addWidget(label, b + 2, 1, 1, 1)
        btn.setEnabled(False)
        b += 1
    parent.viewbtns.setExclusive(True)
    slider = RangeSlider(parent)
    slider.setMinimum(0)
    slider.setMaximum(255)
    slider.setLow(0)
    slider.setHigh(255)
    slider.setTickPosition(QSlider.TicksBelow)
    parent.l0.addWidget(slider, 3, 1, len(parent.view_names) - 2, 1)

    b += 2
    return b


def init_views(parent):
    """ make views using parent.ops

    views in order:
        "Q: ROIs",
        "W: mean img",
        "E: mean img (enhanced)",
        "R: correlation map",
        "T: max projection",
        "Y: mean img chan2, corr",
        "U: mean img chan2",

    assigns parent.views

    """
    parent.Ly, parent.Lx = parent.ops["Ly"], parent.ops["Lx"]
    parent.views = np.zeros((7, parent.Ly, parent.Lx, 3), np.float32)
    for k in range(7):
        if k == 2:
            if "meanImgE" not in parent.ops:
                meanImgE = registration.highpass_mean_image(parent.ops["meanImg"], 
                                                              parent.ops.get("aspect", 1))
                parent.ops["meanImgE"] = meanImgE
            mimg = parent.ops["meanImgE"]
        elif k == 1:
            img = parent.ops["meanImg"]
            mimg1 = np.percentile(img, 1)
            mimg99 = np.percentile(img, 99)
            img = (img - mimg1) / (mimg99 - mimg1)
            img = np.clip(img, 0, 1)
            mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
            if img.shape[0] != parent.Ly or img.shape[1] != parent.Lx:
                mimg[parent.ops["yrange"][0]:parent.ops["yrange"][1],
                     parent.ops["xrange"][0]:parent.ops["xrange"][1]] = img
            else:
                mimg = img
        elif k == 3:
            if "Vcorr" in parent.ops:
                vcorr = parent.ops["Vcorr"]
                mimg1 = np.percentile(vcorr, 1)
                mimg99 = np.percentile(vcorr, 99)
                vcorr = (vcorr - mimg1) / (mimg99 - mimg1)
                vcorr = np.clip(vcorr, 0, 1)
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
                mimg[parent.ops["yrange"][0]:parent.ops["yrange"][1],
                     parent.ops["xrange"][0]:parent.ops["xrange"][1]] = vcorr
                mimg = np.maximum(0, np.minimum(1, mimg))
            else:
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
        elif k == 4:
            if "max_proj" in parent.ops:
                mproj = parent.ops["max_proj"]
                mimg1 = np.percentile(mproj, 1)
                mimg99 = np.percentile(mproj, 99)
                mproj = (mproj - mimg1) / (mimg99 - mimg1)
                mimg = np.zeros((parent.Ly, parent.Lx), np.float32)
                try:
                    mimg[parent.ops["yrange"][0]:parent.ops["yrange"][1],
                         parent.ops["xrange"][0]:parent.ops["xrange"][1]] = mproj
                except:
                    print("maxproj not in combined view")
                mimg = np.maximum(0, np.minimum(1, mimg))
            else:
                mimg = 0.5 * np.ones((parent.Ly, parent.Lx), np.float32)
        elif k == 5:
            if "meanImg_chan2_corrected" in parent.ops:
                mimg = parent.ops["meanImg_chan2_corrected"]
                mimg1 = np.percentile(mimg, 1)
                mimg99 = np.percentile(mimg, 99)
                mimg = (mimg - mimg1) / (mimg99 - mimg1)
                mimg = np.maximum(0, np.minimum(1, mimg))
        elif k == 6:
            if "meanImg_chan2" in parent.ops:
                mimg = parent.ops["meanImg_chan2"]
                mimg1 = np.percentile(mimg, 1)
                mimg99 = np.percentile(mimg, 99)
                mimg = (mimg - mimg1) / (mimg99 - mimg1)
                mimg = np.maximum(0, np.minimum(1, mimg))
        else:
            mimg = np.zeros((parent.Ly, parent.Lx), np.float32)

        mimg *= 255
        mimg = mimg.astype(np.uint8)
        parent.views[k] = np.tile(mimg[:, :, np.newaxis], (1, 1, 3))


def plot_views(parent):
    """ set parent.view1 and parent.view2 image based on parent.ops_plot["view"]"""
    k = parent.ops_plot["view"]
    parent.view1.setImage(parent.views[k], levels=parent.ops_plot["saturation"])
    parent.view2.setImage(parent.views[k], levels=parent.ops_plot["saturation"])
    parent.view1.show()
    parent.view2.show()


class ViewButton(QPushButton):
    """ custom QPushButton class for quadrant plotting
        requires buttons to put into a QButtonGroup (parent.viewbtns)
         allows only 1 button to pressed at a time
    """

    def __init__(self, bid, Text, parent=None):
        super(ViewButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        for b in range(len(parent.views)):
            if parent.viewbtns.button(b).isEnabled():
                parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.ops_plot["view"] = bid
        parent.update_plot()


class RangeSlider(QSlider):
    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the
        exception of valueChanged

        Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
        and modified it
    """

    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = SC_NONE
        self.hover_control = SC_NONE
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Vertical)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet("""
                QSlider::groove:vertical {
                    background: rgb(80, 80, 80);
                    width: 6px;
                }
                QSlider::handle:vertical {
                    background-color: white;
                    border: 1px solid black;
                    height: 8px;
                    width: 18px;
                    margin: 0 -6px;
                }
                """)

        #self.opt = QStyleOptionSlider()
        #self.opt.orientation=QtCore.Qt.Vertical
        #self.initStyleOption(self.opt)
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def level_change(self):
        if self.parent is not None:
            if self.parent.loaded:
                self.parent.ops_plot["saturation"] = [self._low, self._high]
                self.parent.update_plot()

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QPainter(self)
        try:
            style = QApplication.style()
            no_ticks = _slider_no_ticks()

            for i, value in enumerate([self._low, self._high]):
                opt = QStyleOptionSlider()
                self.initStyleOption(opt)

                # Only draw the groove for the first slider so it doesn"t get drawn
                # on top of the existing ones every time
                if i == 0:
                    opt.subControls = SC_SLIDER_GROOVE | SC_SLIDER_HANDLE
                else:
                    opt.subControls = SC_SLIDER_HANDLE

                if self.tickPosition() != no_ticks:
                    opt.subControls |= SC_SLIDER_TICKMARKS

                if self.pressed_control:
                    opt.activeSubControls = self.pressed_control
                    opt.state |= STATE_SUNKEN
                else:
                    opt.activeSubControls = self.hover_control

                opt.sliderPosition = value
                opt.sliderValue = value
                style.drawComplexControl(CC_SLIDER, opt, painter, self)
        finally:
            painter.end()

    def mousePressEvent(self, event):
        event.accept()

        style = QApplication.style()
        button = event.button()
        # In a normal slider control, when the user clicks on a point in the
        # slider"s total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts
        if button:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(CC_SLIDER, opt, event.pos(),
                                                  self)
                if hit == SC_SLIDER_HANDLE:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(SLIDER_MOVE)
                    self.setRepeatAction(SLIDER_NO_ACTION)
                    self.setSliderDown(True)

                    break

            if self.active_slider < 0:
                self.pressed_control = SC_SLIDER_HANDLE
                click_value = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                if abs(click_value - self._low) <= abs(click_value - self._high):
                    self.active_slider = 0
                    self._low = min(click_value, self._high - 1)
                else:
                    self.active_slider = 1
                    self._high = max(click_value, self._low + 1)
                self.click_offset = click_value
                self.triggerAction(SLIDER_MOVE)
                self.setRepeatAction(SLIDER_NO_ACTION)
                self.setSliderDown(True)
                self.update()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != SC_SLIDER_HANDLE:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.level_change()
        self.pressed_control = SC_NONE
        self.hover_control = SC_NONE
        self.setSliderDown(False)
        event.accept()

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()

    def __pixelPosToRangeValue(self, pos):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QApplication.style()

        gr = style.subControlRect(CC_SLIDER, opt, SC_SLIDER_GROOVE, self)
        sr = style.subControlRect(CC_SLIDER, opt, SC_SLIDER_HANDLE, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos - slider_min, slider_max - slider_min,
                                             opt.upsideDown)
