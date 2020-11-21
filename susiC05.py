# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SusiC.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5
import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from susi83 import run_susi
import susi_io
import susi_para
import susi_utils
import pickle
import os

class Ui_MainWindowSusi(object):
    def setupUi(self, MainWindowSusi):
        MainWindowSusi.setObjectName("MainWindowSusi")
        MainWindowSusi.resize(1145, 815)
        self.centralwidget = QtWidgets.QWidget(MainWindowSusi)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setMinimumSize(QtCore.QSize(3, 7))
        self.tabWidget.setBaseSize(QtCore.QSize(3, 7))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.graphicsViewSusi = QtWidgets.QGraphicsView(self.tab_3)
        self.graphicsViewSusi.setGeometry(QtCore.QRect(50, 90, 691, 491))
        self.graphicsViewSusi.setObjectName("graphicsViewSusi")
        self.label = QtWidgets.QLabel(self.tab_3)
        self.label.setGeometry(QtCore.QRect(60, 610, 191, 16))
        self.label.setObjectName("label")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tableWidgetPeat = QtWidgets.QTableWidget(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.tableWidgetPeat.sizePolicy().hasHeightForWidth())
        self.tableWidgetPeat.setSizePolicy(sizePolicy)
        self.tableWidgetPeat.setBaseSize(QtCore.QSize(3, 7))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.tableWidgetPeat.setFont(font)
        self.tableWidgetPeat.setRowCount(2)
        self.tableWidgetPeat.setColumnCount(8)
        self.tableWidgetPeat.setObjectName("tableWidgetPeat")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetPeat.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetPeat.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetPeat.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetPeat.setHorizontalHeaderItem(3, item)
        self.gridLayout_4.addWidget(self.tableWidgetPeat, 0, 1, 1, 1)
        self.groupBoxApriori = QtWidgets.QGroupBox(self.tab_4)
        self.groupBoxApriori.setObjectName("groupBoxApriori")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBoxApriori)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButtonWeather = QtWidgets.QPushButton(self.groupBoxApriori)
        self.pushButtonWeather.setObjectName("pushButtonWeather")
        self.horizontalLayout_2.addWidget(self.pushButtonWeather)
        self.pushButtonMotti = QtWidgets.QPushButton(self.groupBoxApriori)
        self.pushButtonMotti.setObjectName("pushButtonMotti")
        self.horizontalLayout_2.addWidget(self.pushButtonMotti)
        self.gridLayout_4.addWidget(self.groupBoxApriori, 1, 0, 1, 1)
        self.tableWidgetScenario = QtWidgets.QTableWidget(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.tableWidgetScenario.sizePolicy().hasHeightForWidth())
        self.tableWidgetScenario.setSizePolicy(sizePolicy)
        self.tableWidgetScenario.setBaseSize(QtCore.QSize(3, 7))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.tableWidgetScenario.setFont(font)
        self.tableWidgetScenario.setRowCount(4)
        self.tableWidgetScenario.setColumnCount(2)
        self.tableWidgetScenario.setObjectName("tableWidgetScenario")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetScenario.setItem(3, 1, item)
        self.gridLayout_4.addWidget(self.tableWidgetScenario, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tableWidgetResults = QtWidgets.QTableWidget(self.tab_2)
        self.tableWidgetResults.setGeometry(QtCore.QRect(20, 30, 651, 631))
        self.tableWidgetResults.setBaseSize(QtCore.QSize(4, 9))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.tableWidgetResults.setFont(font)
        self.tableWidgetResults.setRowCount(13)
        self.tableWidgetResults.setColumnCount(4)
        self.tableWidgetResults.setObjectName("tableWidgetResults")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetResults.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(10, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(11, 0, item)
        item = QtWidgets.QTableWidgetItem()
        brush = QtGui.QBrush(QtGui.QColor(232, 224, 113))
        brush.setStyle(QtCore.Qt.Dense4Pattern)
        item.setBackground(brush)
        self.tableWidgetResults.setItem(12, 0, item)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.textEditTeam = QtWidgets.QTextEdit(self.tab)
        self.textEditTeam.setGeometry(QtCore.QRect(30, 50, 291, 561))
        self.textEditTeam.setObjectName("textEditTeam")
        self.tabWidget.addTab(self.tab, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindowSusi.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindowSusi)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1145, 26))
        self.menubar.setObjectName("menubar")
        MainWindowSusi.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindowSusi)
        self.statusbar.setObjectName("statusbar")
        MainWindowSusi.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindowSusi)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBoxSimulation = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.groupBoxSimulation.setObjectName("groupBoxSimulation")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBoxSimulation)
        self.gridLayout.setObjectName("gridLayout")
        self.label_start = QtWidgets.QLabel(self.groupBoxSimulation)
        self.label_start.setObjectName("label_start")
        self.gridLayout.addWidget(self.label_start, 0, 0, 1, 1)
        self.spinBoxStartYr = QtWidgets.QSpinBox(self.groupBoxSimulation)
        self.spinBoxStartYr.setObjectName("spinBoxStartYr")
        self.gridLayout.addWidget(self.spinBoxStartYr, 1, 0, 1, 1)
        self.label_end_year = QtWidgets.QLabel(self.groupBoxSimulation)
        self.label_end_year.setObjectName("label_end_year")
        self.gridLayout.addWidget(self.label_end_year, 2, 0, 1, 1)
        self.spinBoxEndYr = QtWidgets.QSpinBox(self.groupBoxSimulation)
        self.spinBoxEndYr.setObjectName("spinBoxEndYr")
        self.gridLayout.addWidget(self.spinBoxEndYr, 3, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBoxSimulation, 0, 0, 1, 1)
        self.groupBoxSite = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.groupBoxSite.setObjectName("groupBoxSite")
        self.formLayout_2 = QtWidgets.QFormLayout(self.groupBoxSite)
        self.formLayout_2.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_stand_age = QtWidgets.QLabel(self.groupBoxSite)
        self.label_stand_age.setObjectName("label_stand_age")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_stand_age)
        self.lineEditAge = QtWidgets.QLineEdit(self.groupBoxSite)
        self.lineEditAge.setObjectName("lineEditAge")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.lineEditAge)
        self.labelStripWidth = QtWidgets.QLabel(self.groupBoxSite)
        self.labelStripWidth.setObjectName("labelStripWidth")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.labelStripWidth)
        self.lineEditStripWidth = QtWidgets.QLineEdit(self.groupBoxSite)
        self.lineEditStripWidth.setObjectName("lineEditStripWidth")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.lineEditStripWidth)
        self.labelSitePara = QtWidgets.QLabel(self.groupBoxSite)
        self.labelSitePara.setObjectName("labelSitePara")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.labelSitePara)
        self.comboBoxSitePara = QtWidgets.QComboBox(self.groupBoxSite)
        self.comboBoxSitePara.setObjectName("comboBoxSitePara")
        self.comboBoxSitePara.addItem("")
        self.comboBoxSitePara.addItem("")
        self.comboBoxSitePara.addItem("")
        self.comboBoxSitePara.addItem("")
        self.comboBoxSitePara.addItem("")
        self.comboBoxSitePara.addItem("")
        self.comboBoxSitePara.addItem("")
        self.comboBoxSitePara.addItem("")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.SpanningRole, self.comboBoxSitePara)
        self.lineEditLogPrice = QtWidgets.QLineEdit(self.groupBoxSite)
        self.lineEditLogPrice.setObjectName("lineEditLogPrice")
        self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.lineEditLogPrice)
        self.label_2 = QtWidgets.QLabel(self.groupBoxSite)
        self.label_2.setObjectName("label_2")
        self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(self.groupBoxSite)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.lineEditPulpPrice = QtWidgets.QLineEdit(self.groupBoxSite)
        self.lineEditPulpPrice.setObjectName("lineEditPulpPrice")
        self.formLayout_2.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.lineEditPulpPrice)
        self.label_4 = QtWidgets.QLabel(self.groupBoxSite)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.lineEditDitchCost = QtWidgets.QLineEdit(self.groupBoxSite)
        self.lineEditDitchCost.setObjectName("lineEditDitchCost")
        self.formLayout_2.setWidget(12, QtWidgets.QFormLayout.LabelRole, self.lineEditDitchCost)
        self.gridLayout_3.addWidget(self.groupBoxSite, 1, 0, 1, 1)
        self.groupBoxRun = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.groupBoxRun.setObjectName("groupBoxRun")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBoxRun)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButtonRunSusi = QtWidgets.QPushButton(self.groupBoxRun)
        self.pushButtonRunSusi.setObjectName("pushButtonRunSusi")
        self.horizontalLayout.addWidget(self.pushButtonRunSusi)
        self.gridLayout_3.addWidget(self.groupBoxRun, 3, 0, 1, 1)
        self.groupBoxFiles = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.groupBoxFiles.setObjectName("groupBoxFiles")
        self.formLayout = QtWidgets.QFormLayout(self.groupBoxFiles)
        self.formLayout.setObjectName("formLayout")
        self.pushButtonHaeParametriTiedosto = QtWidgets.QPushButton(self.groupBoxFiles)
        self.pushButtonHaeParametriTiedosto.setObjectName("pushButtonHaeParametriTiedosto")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.pushButtonHaeParametriTiedosto)
        self.lineEditMottiTiedosto = QtWidgets.QLineEdit(self.groupBoxFiles)
        self.lineEditMottiTiedosto.setObjectName("lineEditMottiTiedosto")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEditMottiTiedosto)
        self.pushButtonGetWeatherFile = QtWidgets.QPushButton(self.groupBoxFiles)
        self.pushButtonGetWeatherFile.setObjectName("pushButtonGetWeatherFile")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.pushButtonGetWeatherFile)
        self.lineEditWeatherFile = QtWidgets.QLineEdit(self.groupBoxFiles)
        self.lineEditWeatherFile.setObjectName("lineEditWeatherFile")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEditWeatherFile)
        self.pushButtonGeWorkFolder = QtWidgets.QPushButton(self.groupBoxFiles)
        self.pushButtonGeWorkFolder.setObjectName("pushButtonGeWorkFolder")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.pushButtonGeWorkFolder)
        self.lineEditWorkFolder = QtWidgets.QLineEdit(self.groupBoxFiles)
        self.lineEditWorkFolder.setObjectName("lineEditWorkFolder")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEditWorkFolder)
        self.gridLayout_3.addWidget(self.groupBoxFiles, 2, 0, 1, 1)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindowSusi.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)

        self.retranslateUi(MainWindowSusi)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindowSusi)

#----Own functionalities from here
        scene = QtWidgets.QGraphicsScene()      
        scene.addPixmap(PyQt5.QtGui.QPixmap('susi2.png'))
        self.graphicsViewSusi.setScene(scene)
        self.graphicsViewSusi.scale(0.5,0.45)                     
        
        self.pushButtonRunSusi.clicked.connect(self.call_susi)
        self.pushButtonHaeParametriTiedosto.clicked.connect(self.get_motti_file)
        self.pushButtonGetWeatherFile.clicked.connect(self.get_weather_file)
        self.pushButtonGeWorkFolder.clicked.connect(self.get_work_folder)
        self.pushButtonWeather.clicked.connect(self.draw_weather)        
        self.pushButtonMotti.clicked.connect(self.draw_motti)        
        
        #initialization
        self.spinBoxStartYr.setMinimum(1961)
        self.spinBoxStartYr.setMaximum(2050)
        self.spinBoxEndYr.setMinimum(1961)
        self.spinBoxEndYr.setMaximum(2050)
        
        self.spinBoxStartYr.setValue(2010)
        self.spinBoxEndYr.setValue(2014)
        
        ageSim=  float(self.lineEditAge.text()) #50. #idata.T[nro]['age_ini']                                                                     #90
        sarkaSim = float(self.lineEditStripWidth.text()) #40. #idata.T[nro]['stripw']  
        n = int(sarkaSim / 2)
        dd=[]
        scen =[]
        for k in range(self.tableWidgetScenario.rowCount()):
            scen.append(self.tableWidgetScenario.item(k, 0).text())
            dd.append(float(self.tableWidgetScenario.item(k, 1).text()) / -100.)
        
        self.sitetype = self.comboBoxSitePara.currentText()
        self.spara, self.outpara = self.get_ini_para(folderName=None, hdomSim=None, volSim=None, scen = scen,
                      ageSim=ageSim, sarkaSim=sarkaSim, ddwest=dd, ddeast=dd, n=n)


        #itemlist = self.spara.keys()
        #itemlist.append('previous simulation')
        #self.comboBoxSitePara.addItems(itemlist)        
        self.comboBoxSitePara.setCurrentIndex(0)
        self.comboBoxSitePara.currentIndexChanged.connect(self.show_site_para)
        #self.comboBoxSitePara.setToolTip('Predefined peat profile parameters')        
        self.show_outpara(self.outpara)
        self.ini_para()
        self.print_team()





    def retranslateUi(self, MainWindowSusi):
        _translate = QtCore.QCoreApplication.translate
        MainWindowSusi.setWindowTitle(_translate("MainWindowSusi", "Susi - Suosimulaattori 2020"))
        self.label.setText(_translate("MainWindowSusi", "Susi versio 2.0 Kesäkuu 2020"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindowSusi", "Aloitus"))
        item = self.tableWidgetPeat.horizontalHeaderItem(0)
        item.setText(_translate("MainWindowSusi", "Syvyys [m]"))
        item = self.tableWidgetPeat.horizontalHeaderItem(1)
        item.setText(_translate("MainWindowSusi", "Turvelaji"))
        item = self.tableWidgetPeat.horizontalHeaderItem(2)
        item.setText(_translate("MainWindowSusi", "von Post"))
        item = self.tableWidgetPeat.horizontalHeaderItem(3)
        item.setText(_translate("MainWindowSusi", "Tiheys"))
        self.groupBoxApriori.setTitle(_translate("MainWindowSusi", "Kuvat"))
        self.pushButtonWeather.setText(_translate("MainWindowSusi", "Piirrä säädata"))
        self.pushButtonMotti.setText(_translate("MainWindowSusi", "Piirrä metsikkö"))
        item = self.tableWidgetScenario.horizontalHeaderItem(0)
        item.setText(_translate("MainWindowSusi", "Skenaario"))
        item = self.tableWidgetScenario.horizontalHeaderItem(1)
        item.setText(_translate("MainWindowSusi", "Ojan syvyys [cm]"))
        __sortingEnabled = self.tableWidgetScenario.isSortingEnabled()
        self.tableWidgetScenario.setSortingEnabled(False)
        item = self.tableWidgetScenario.item(0, 0)
        item.setText(_translate("MainWindowSusi", "Skenaario 1"))
        item = self.tableWidgetScenario.item(0, 1)
        item.setText(_translate("MainWindowSusi", "30"))
        item = self.tableWidgetScenario.item(1, 0)
        item.setText(_translate("MainWindowSusi", "Skenaario 2"))
        item = self.tableWidgetScenario.item(1, 1)
        item.setText(_translate("MainWindowSusi", "50"))
        item = self.tableWidgetScenario.item(2, 0)
        item.setText(_translate("MainWindowSusi", "Skenaario 3"))
        item = self.tableWidgetScenario.item(2, 1)
        item.setText(_translate("MainWindowSusi", "70"))
        item = self.tableWidgetScenario.item(3, 0)
        item.setText(_translate("MainWindowSusi", "Skenaario 4"))
        item = self.tableWidgetScenario.item(3, 1)
        item.setText(_translate("MainWindowSusi", "90"))
        self.tableWidgetScenario.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindowSusi", "Input"))
        item = self.tableWidgetResults.verticalHeaderItem(0)
        item.setText(_translate("MainWindowSusi", "Pohjavesi, kesä [m]"))
        item = self.tableWidgetResults.verticalHeaderItem(1)
        item.setText(_translate("MainWindowSusi", "Pohjavesi: muutos [m]"))
        item = self.tableWidgetResults.verticalHeaderItem(2)
        item.setText(_translate("MainWindowSusi", "Tilavuus, alku  [m3 ha-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(3)
        item.setText(_translate("MainWindowSusi", "Tilavuus, loppu [m3 ha-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(4)
        item.setText(_translate("MainWindowSusi", "Tuotos [m3 ha-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(5)
        item.setText(_translate("MainWindowSusi", "Tuotos,  muutos [m3 ha-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(6)
        item.setText(_translate("MainWindowSusi", "Kasvu [m3 ha-1 a-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(7)
        item.setText(_translate("MainWindowSusi", "Kasvu, muutos [m3 ha-1 a-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(8)
        item.setText(_translate("MainWindowSusi", "Tukki, loppu [m3 ha-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(9)
        item.setText(_translate("MainWindowSusi", "Tukki,: muutos [m3 ha-1]"))
        item = self.tableWidgetResults.verticalHeaderItem(10)
        item.setText(_translate("MainWindowSusi", "NPV [€ ha-1] i=2%"))
        item = self.tableWidgetResults.verticalHeaderItem(11)
        item.setText(_translate("MainWindowSusi", "NPV [€ ha-1] i=3%"))
        item = self.tableWidgetResults.verticalHeaderItem(12)
        item.setText(_translate("MainWindowSusi", "NPV [€ ha-1] i=4%"))
        item = self.tableWidgetResults.horizontalHeaderItem(0)
        item.setText(_translate("MainWindowSusi", "Skenaario 1"))
        item = self.tableWidgetResults.horizontalHeaderItem(1)
        item.setText(_translate("MainWindowSusi", "Skenaario 2"))
        item = self.tableWidgetResults.horizontalHeaderItem(2)
        item.setText(_translate("MainWindowSusi", "Skenaario 3"))
        item = self.tableWidgetResults.horizontalHeaderItem(3)
        item.setText(_translate("MainWindowSusi", "Skenaario 4"))
        __sortingEnabled = self.tableWidgetResults.isSortingEnabled()
        self.tableWidgetResults.setSortingEnabled(False)
        self.tableWidgetResults.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindowSusi", "Tulokset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindowSusi", "Työryhmä"))
        self.groupBoxSimulation.setTitle(_translate("MainWindowSusi", "Simulaatio"))
        self.label_start.setText(_translate("MainWindowSusi", "Aloitusvuosi"))
        self.label_end_year.setText(_translate("MainWindowSusi", "Lopetusvuosi"))
        self.groupBoxSite.setTitle(_translate("MainWindowSusi", "Metsikkö ja kasvupaikka"))
        self.label_stand_age.setText(_translate("MainWindowSusi", "Ikä, vuotta"))
        self.lineEditAge.setText(_translate("MainWindowSusi", "40"))
        self.labelStripWidth.setText(_translate("MainWindowSusi", "Sarkaleveys, m"))
        self.lineEditStripWidth.setText(_translate("MainWindowSusi", "40"))
        self.labelSitePara.setText(_translate("MainWindowSusi", "Kasvupaikka"))
        self.comboBoxSitePara.setItemText(0, _translate("MainWindowSusi", "Ruohoturvekangas, rahkaturve"))
        self.comboBoxSitePara.setItemText(1, _translate("MainWindowSusi", "Ruohoturvekangas, muu turve"))
        self.comboBoxSitePara.setItemText(2, _translate("MainWindowSusi", "Mustikkaturvekangas, rahkaturve"))
        self.comboBoxSitePara.setItemText(3, _translate("MainWindowSusi", "Mustikkaturvekangas, muu turve"))
        self.comboBoxSitePara.setItemText(4, _translate("MainWindowSusi", "Puolukkaturvekangas, rahkaturve"))
        self.comboBoxSitePara.setItemText(5, _translate("MainWindowSusi", "Puolukkaturvekangas, muu turve"))
        self.comboBoxSitePara.setItemText(6, _translate("MainWindowSusi", "Varputurvekangas, rahkaturve"))
        self.comboBoxSitePara.setItemText(7, _translate("MainWindowSusi", "Varputurvekangas, muu turve"))
        self.lineEditLogPrice.setText(_translate("MainWindowSusi", "57.9"))
        self.label_2.setText(_translate("MainWindowSusi", "Tukin hinta € m-3"))
        self.label_3.setText(_translate("MainWindowSusi", "Kuitupuun hinta € m-3"))
        self.lineEditPulpPrice.setText(_translate("MainWindowSusi", "20.1"))
        self.label_4.setText(_translate("MainWindowSusi", "Ojituskustannus € ha-1"))
        self.lineEditDitchCost.setText(_translate("MainWindowSusi", "200.0"))
        self.groupBoxRun.setTitle(_translate("MainWindowSusi", "Laske"))
        self.pushButtonRunSusi.setText(_translate("MainWindowSusi", "Laske"))
        self.groupBoxFiles.setTitle(_translate("MainWindowSusi", "Files"))
        self.pushButtonHaeParametriTiedosto.setText(_translate("MainWindowSusi", "Motti tiedosto"))
        self.pushButtonGetWeatherFile.setText(_translate("MainWindowSusi", "Säädata"))
        self.pushButtonGeWorkFolder.setText(_translate("MainWindowSusi", "Tulostiedosto"))




    def call_susi(self):
        mottifile =  str(self.lineEditMottiTiedosto.text()) 
        ageSim= float(self.lineEditAge.text()) 
        folderName= str(self.lineEditWorkFolder.text())  #'C:\Apps\WinPython-64bit-2.7.10.3\susi_5_1\outputs\\'
        #susiPath = 'C:\Apps\WinPython-64bit-2.7.10.3\susi_5_1\\'
        susiPath = os.getcwd()
        self.outpara['outfolder']=folderName + '\\'
        wdata= str(self.lineEditWeatherFile.text())  #'C:\Apps\WinPython-64bit-2.7.10.3\susi_5_1\\wfiles\\Lohja_weather.csv'  
        syr = int(self.spinBoxStartYr.value())
        eyr = int(self.spinBoxEndYr.value())        
        start_date = datetime.datetime(syr,1,1); end_date=datetime.datetime(eyr,12,31)        

        forc=susi_utils.read_FMI_weather(0, start_date, end_date, sourcefile=wdata)           # read weather input        print ('done')
        site = str(self.comboBoxSitePara.currentText()) 
        strip_w= float(self.lineEditStripWidth.text())
        
        ini_para={}
        ini_para['start_yr']=self.spinBoxStartYr.value()
        ini_para['end_yr']=self.spinBoxEndYr.value()
        ini_para['sitetype']= self.comboBoxSitePara.currentText() #self.spara['sitetype']['sfc'] #2 #Here Mtkg....Ptkg
        ini_para['iniage']=float(self.lineEditAge.text())
        ini_para['strip_width']=float(self.lineEditStripWidth.text())
            
        ini_para['motti_file']=self.lineEditMottiTiedosto.text()
        ini_para['weather_file']=self.lineEditWeatherFile.text()
        ini_para['work_folder']=self.lineEditWorkFolder.text()
        ini_para['site']=site
        strip_w=float(self.lineEditStripWidth.text())        
        self.compose_para()

        
        self.spara[site]['L']=strip_w
        
        with open(susiPath+'\\paras.pickle', 'wb') as handle:
            pickle.dump(self.spara[site], handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(susiPath+'\\inipara.pickle', 'wb') as handle:
            pickle.dump(ini_para, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #run_susi(forc, self.spara[site], self.outpara, syr, eyr, wlocation = 'undefined', mottifile=mottifile, peat= 'other', photosite='All data', 
        #         folderName=folderName, vonP_lyr='vp27', ageSim=ageSim, sarkaSim=strip_w, sfc=sfc, susiPath=susiPath)


        kaista = 1 #idata.T[nro]['kaista']

        ageSim=  float(self.lineEditAge.text()) #50. #idata.T[nro]['age_ini']                                                                     #90
        sarkaSim = float(self.lineEditStripWidth.text()) #40. #idata.T[nro]['stripw']  
        dd=[]
        scen =[]
        for k in range(10):
            try: 
                scen.append(self.tableWidgetScenario.item(k, 0).text())
                dd.append(float(self.tableWidgetScenario.item(k, 1).text()) / -100.)
            except:
                pass
        self.spara[site]['scen'] = scen
        self.spara[site]['ditch depth west'] = dd
        self.spara[site]['ditch depth east'] = dd
        self.spara[site]['ditch depth 20y west'] = dd                                           #ojan syvyys 20 vuotta simuloinnin aloituksesta
        self.spara[site]['ditch depth 20y east'] = dd                                           #ojan syvyys 20 vuotta simuloinnin aloituksesta

        wpara, cpara, org_para, photopara =  self.get_susi_para(wlocation='undefined', photosite='All data', susiPath = susiPath)
      
        #print ('********************')
        #print (self.spara[site])
        #print ('********************')
        
        #v_ini, v, iv, cb, dcb, w, dw, logs, pulp, dv, dlogs, dpulp, yrs = run_susi(forc, wpara, cpara, 
        #                org_para, self.spara[site], self.outpara, photopara, syr, eyr, wlocation = 'undefined', 
        #                mottifile=mottifile, peat= 'other', photosite='All data', 
        #                folderName=folderName,ageSim=ageSim, sarkaSim=sarkaSim, sfc=self.spara[site]['sfc'], susiPath=susiPath, kaista=kaista)

        v_ini, v, iv, cbt, dcbt, cb, dcb,  w, dw, logs, pulp, dv, dlogs, dpulp, yrs, bmgr,Nleach, \
                        Pleach, Kleach, DOCleach, runoff = run_susi(forc, wpara, cpara, 
                        org_para, self.spara[site], self.outpara, photopara, syr, eyr, wlocation = 'undefined', 
                        mottifile=mottifile, peat= 'other', photosite='All data', 
                        folderName=folderName,ageSim=ageSim, sarkaSim=sarkaSim, sfc=self.spara[site]['sfc'], susiPath=susiPath, kaista=kaista)
  
        
        print ('now returned')
        
        #print (v_ini, v, iv, cb, dcb, w, dw, logs, pulp, dv, dlogs, dpulp, yrs)
        print (v_ini, v, iv, cb, dcb, cbt, dcbt, w, dw, logs, pulp, dv, dlogs, dpulp, yrs)
  
        yi = v-v_ini
        rounds = len(v)   #number of scenarios
        p_log = float(self.lineEditLogPrice.text())
        p_pulp = float(self.lineEditPulpPrice.text())
        p_ditch = float(self.lineEditDitchCost.text())
        npv2 = p_log* dlogs * np.exp(-0.02*yrs) + p_pulp* dpulp * np.exp(-0.02*yrs) - p_ditch
        npv3 = p_log* dlogs * np.exp(-0.03*yrs) + p_pulp* dpulp * np.exp(-0.03*yrs) - p_ditch
        npv4 = p_log* dlogs * np.exp(-0.04*yrs) + p_pulp* dpulp * np.exp(-0.04*yrs) - p_ditch

        print ('npv', npv2)
        
        for k in range(self.tableWidgetResults.columnCount()): 
            try:    
                self.tableWidgetResults.setItem(0,k,QtWidgets.QTableWidgetItem(str(np.round(w[k],2))))        #pohjavesi
                if k>0: self.tableWidgetResults.setItem(1,k,QtWidgets.QTableWidgetItem(str(np.round(w[k]-w[0],2))))   #pohjavesi, muutos
                self.tableWidgetResults.setItem(2,k,QtWidgets.QTableWidgetItem(str(np.round(v_ini,2))))       #tilavuus, alku
                self.tableWidgetResults.setItem(3,k,QtWidgets.QTableWidgetItem(str(np.round(v[k],2))))        #tilavuus, loppu    
                self.tableWidgetResults.setItem(4,k,QtWidgets.QTableWidgetItem(str(np.round(yi[k],2))))       #tuotos
                if k>0: self.tableWidgetResults.setItem(5,k,QtWidgets.QTableWidgetItem(str(np.round(yi[k]-yi[0],2))))   #tuotos, muutos
                self.tableWidgetResults.setItem(6,k,QtWidgets.QTableWidgetItem(str(np.round((yi[k]/(eyr-syr + 1)),2))))   #kasvu
                if k>0: self.tableWidgetResults.setItem(7,k,QtWidgets.QTableWidgetItem(str(np.round((yi[k]-yi[0])/(eyr-syr + 1),2))))   #kasvu, muutos
                
                self.tableWidgetResults.setItem(8,k,QtWidgets.QTableWidgetItem(str(np.round(logs[k],2))))     #tukki lopussa
                if k>0: self.tableWidgetResults.setItem(9,k,QtWidgets.QTableWidgetItem(str(np.round(logs[k]-logs[0],2))))

                if k>0: self.tableWidgetResults.setItem(10,k,QtWidgets.QTableWidgetItem(str(np.round(npv2[k],2))))
                if k>0: self.tableWidgetResults.setItem(11,k,QtWidgets.QTableWidgetItem(str(np.round(npv3[k],2))))
                if k>0: self.tableWidgetResults.setItem(12,k,QtWidgets.QTableWidgetItem(str(np.round(npv4[k],2))))
                
            except:
                pass

            R, G, B = 227,236,104   
            for k in range(13):
                self.tableWidgetResults.item(k, 0).setBackground(QtGui.QColor(R,G,B))

        outf = folderName + '/susi_tulokset.xlsx'
        #tulokset 
        gr =[yi[k]/(eyr-syr + 1) for k in range(4)]
        dgr =[(yi[k]-yi[0])/(eyr-syr + 1) for k in range(4)]
        df = pd.DataFrame(list(zip(w, dw, v, dv, yi, gr, dgr, npv2, npv3, npv4)), index=['Simulation 1','Simulation 2','Simulation 3','Simulation 4'],
                          columns=['Pohjavesi', 'PV muutos', 'Tilavuus', 'Tilavuus muutos', 'Tuotos', 'Kasvu', 'Kasvu muutos', 'NPV2', 'NPV3', 'NPV4'])
        df = df.T
        df.to_excel(outf)
        dyi= [yi[k]-yi[0] for k in range(4)]
        dg = [(yi[k]-yi[0])/(eyr-syr + 1) for k in range(4)]
        fs=12
        fig = plt.figure(num='Susi - skenaariot', figsize=[15.,8.], facecolor='#C1ECEC')  #see hex color codes from https://www.rapidtables.com/web/color/html-color-codes.html
        plt.subplot(221)
        plt.plot(-np.array(dd), np.array(w))
        plt.xlabel('Ojan syvyys [m]')
        plt.ylabel('Pohaveden syvyys, kesä [m]')
        plt.subplot(222)
        plt.plot(-np.array(dd), np.array(dyi))
        plt.xlabel('Ojan syvyys [m]')
        plt.ylabel('Lisätuotos [$m^{3}$ $ha^{-1}$]')
        plt.subplot(223)
        plt.plot(-np.array(dd), np.array(dg))
        plt.xlabel('Ojan syvyys [m]')
        plt.ylabel('Lisäkasvu [$m^{3}$ $ha^{-1}$ $a^{-1}$]')
        plt.subplot(224)
        plt.plot(-np.array(dd), np.array(npv2), label='i=2%')
        plt.plot(-np.array(dd), np.array(npv3), label='i=3%')
        plt.plot(-np.array(dd), np.array(npv4), label='i=4%')
        plt.xlabel('Ojan syvyys [m]')
        plt.ylabel('Nettonykyarvo [€ $ha^{-1}$]')
        plt.legend(loc='best')
        plt.show()        
        
    def get_motti_file(self):
        self.mottifile, _ = QtWidgets.QFileDialog.getOpenFileName(caption="Get Motti-simulation file", filter = "*.xls")
        self.lineEditMottiTiedosto.clear()
        self.lineEditMottiTiedosto.setText(self.mottifile)        

    def get_weather_file(self):
        self.weatherfile, _ = QtWidgets.QFileDialog.getOpenFileName(caption="Get weather file", filter = "*.csv")
        #self.weatherfile, _ = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        self.lineEditWeatherFile.clear()        
        self.lineEditWeatherFile.setText(self.weatherfile)
        
    def get_work_folder(self):
        self.OutFolder= QtWidgets.QFileDialog.getExistingDirectory(caption="Set work folder")
        self.lineEditWorkFolder.clear()
        self.lineEditWorkFolder.setText(self.OutFolder)  

    def show_site_para(self):        
        site = self.comboBoxSitePara.currentText()
        #self.tableWidgetScenario.clearContents()
        self.tableWidgetPeat.clearContents()
        nscens = len(self.spara[site]['scenario name'])
        self.tableWidgetScenario.setRowCount(max(nscens,10))        
        
        
        #for n, scen in enumerate(self.spara[site]['scenario name']):
        #    self.tableWidgetScenario.setItem(n,0,QtWidgets.QTableWidgetItem(str(scen)))
        #    dd=str(self.spara[site]['ditch depth west'][n]) #+ ' ' + str(self.spara[site]['ditch depth east'][n])           
        #    self.tableWidgetScenario.setItem(n,1,QtWidgets.QTableWidgetItem(dd))
            #dd20y = str(self.spara[site]['ditch depth 20y west'][n]) + ' ' +str(self.spara[site]['ditch depth 20y east'][n])
            #self.tableWidgetScenario.setItem(n,2,QtWidgets.QTableWidgetItem(dd20y))
                
        nlyrs = len(self.spara[site]['peat type'])
        self.tableWidgetPeat.setRowCount(max(nlyrs,20))
        dz = self.spara[site]['dzLyr']
        z = -0.5*dz
        for n, pt in enumerate(self.spara[site]['peat type'][0:5]):
            z = np.round(z + dz, 2)
            self.tableWidgetPeat.setItem(n,0,QtWidgets.QTableWidgetItem(str(z)))
            self.tableWidgetPeat.setItem(n,1,QtWidgets.QTableWidgetItem(str(pt)))
            vp = self.spara[site]['vonP top'][n]
            self.tableWidgetPeat.setItem(n,2,QtWidgets.QTableWidgetItem(str(vp)))            
            bd = self.spara[site]['bd top'][n]
            self.tableWidgetPeat.setItem(n,3,QtWidgets.QTableWidgetItem(str(bd)))            

    def compose_para(self):
        site = self.comboBoxSitePara.currentText()
        sc = []; dd=[] #; dd20y=[]
        for k in range(self.tableWidgetScenario.rowCount()):
            try:            
                sc.append(self.tableWidgetScenario.item(k, 0).text())
                dd.append(float(self.tableWidgetScenario.item(k, 1).text())/-100.)
                #dd20y.append(float(self.tableWidgetScenario.item(k, 2).text()))
            except:
                pass
        pt=[]; vp =[]; bd=[]
        for k in range(self.tableWidgetPeat.rowCount()):
            try:            
                pt.append(self.tableWidgetPeat.item(k, 1).text())
                vp.append(int(self.tableWidgetPeat.item(k, 2).text()))
                bd.append(float(self.tableWidgetPeat.item(k,3).text()))
            except:
                pass
        self.spara[site]['scenario name']= sc
        self.spara[site]['ditch depth']=dd
        self.spara[site]['ditch depth 20y']= dd
        self.spara[site]['peat type']=pt
        self.spara[site]['vopP top']=vp
        self.spara[site]['bd top']=bd
        #self.spara[site]['sfc'] = self.comboBoxSitePara.currentText() #Here Mtkg, Ptkg
        self.spara[site]['age'] = float(self.lineEditAge.text())

    def ini_para(self):
        susiPath = os.getcwd()
        try:
            with open(susiPath+'\\paras.pickle', 'rb') as handle:
                self.spara['previous simulation'] = pickle.load(handle)        
            self.comboBoxSitePara.addItem('aiempi simulaatio')
        except:
            print ('No previous parameterfile in ', susiPath)
        try:
            with open(susiPath+'\\inipara.pickle', 'rb') as handle:
                ini = pickle.load(handle)        
            self.spinBoxStartYr.setValue(int(ini['start_yr']))
            self.spinBoxEndYr.setValue(int(ini['end_yr']))
            self.lineEditAge.setText(str(ini['iniage']))
            self.lineEditWeatherFile.setText(str(ini['weather_file']))
            self.lineEditMottiTiedosto.setText(str(ini['motti_file']))
            self.lineEditWorkFolder.setText(str(ini['work_folder']))
            self.lineEditStripWidth.setText(str(ini['strip_width']))
            self.comboBoxSitePara.setCurrentText(ini['site'])
        except:
            print ('No previous initial values in',  susiPath, '\inipara.pickle')
            
    def show_outpara(self, outpara):        
        pass

        
    def draw_weather(self):
        wdata= str(self.lineEditWeatherFile.text())  #'C:\Apps\WinPython-64bit-2.7.10.3\susi_5_1\\wfiles\\Lohja_weather.csv'  
        syr = int(self.spinBoxStartYr.value())
        eyr = int(self.spinBoxEndYr.value())        
        start_date = datetime.datetime(syr,1,1); end_date=datetime.datetime(eyr,12,31)        
        forc=susi_utils.read_FMI_weather(0, start_date, end_date, sourcefile=wdata)           # read weather input
        susi_io.weather_fig(forc)

    def draw_motti(self):
        mottifile = str(self.lineEditMottiTiedosto.text())        
        syr = int(self.spinBoxStartYr.value())
        eyr = int(self.spinBoxEndYr.value())        

        #sfc = 2   #soil fertility class
        ageSim= float(self.lineEditAge.text()) 
        yrs = float(eyr-syr+1)
        df = susi_utils.get_motti(mottifile)
        susi_io.motti_fig(df, ageSim, yrs)
    
    def get_susi_para(self, wlocation=None, photosite='All data', susiPath = None):
    
        #********** Stand parameters and weather forcing*******************
        #--------------Weather variables 10 km x 10 km grid   
        if susiPath is None: susiPath=""
        wpara ={
    
            'undefined': {
            'infolder': susiPath + '\\wfiles\\',
            'infile_d':'Tammela_weather_1.csv',
            'start_yr': 1980, 'end_yr': 1984, 
            'description': 'Undefined, Finland',
            'lat': 65.00, 'lon': 25.00},
    
            }
    
        cpara = {'dt': 86400.0,
                'flow' : { # flow field
                         'zmeas': 2.0,
                         'zground': 0.5,
                         'zo_ground': 0.01
                         },
                'interc': { # interception
                            'wmax': 0.5, # storage capacity for rain (mm/LAI)
                            'wmaxsnow': 4.0, # storage capacity for snow (mm/LAI),
                            },
                'snow': {
                        # degree-day snow model
                        'kmelt': 2.8934e-05, # melt coefficient in open (mm/s)
                        'kfreeze': 5.79e-6, # freezing coefficient (mm/s)
                        'r': 0.05 # maximum fraction of liquid in snow (-)
                        },
    
                'physpara': {
                            # canopy conductance
                            'amax': 10.0, # maximum photosynthetic rate (umolm-2(leaf)s-1)
                            'g1_conif': 2.1, # stomatal parameter, conifers
                            'g1_decid': 3.5, # stomatal parameter, deciduous
                            'q50': 50.0, # light response parameter (Wm-2)
                            'kp': 0.6, # light attenuation parameter (-)
                            'rw': 0.20, # critical value for REW (-),
                            'rwmin': 0.02, # minimum relative conductance (-)
                            # soil evaporation
                            'gsoil': 1e-2 # soil surface conductance if soil is fully wet (m/s)
                            },
                'phenopara': {
                            #seasonal cycle of physiology: smax [degC], tau[d], xo[degC],fmin[-](residual photocapasity)
                            'smax': 18.5, # degC
                            'tau': 13.0, # days
                            'xo': -4.0, # degC
                            'fmin': 0.05, # minimum photosynthetic capacity in winter (-)
                            },
                'state': {
                           'lai_conif': 3.0, # conifer 1-sided LAI (m2 m-2)
                           'lai_decid_max': 0.01, # maximum annual deciduous 1-sided LAI (m2 m-2): 
                           'hc': 16.0, # canopy height (m)
                           'cf': 0.7, # canopy closure fraction (-)
                           #initial state of canopy storage [mm] and snow water equivalent [mm]
                           'w': 0.0, # canopy storage mm
                           'swe': 0.0, # snow water equivalent mm
                           }
                }
        org_para = {
               'org_depth': 0.04, # depth of organic top layer (m)
               'org_poros': 0.9, # porosity (-)
               'org_fc': 0.3, # field capacity (-)
               'org_rw': 0.24, # critical vol. moisture content (-) for decreasing phase in Ef
               'pond_storage_max': 0.01, # max ponding allowed (m)
               #initial states
               'org_sat': 1.0, # organic top layer saturation ratio (-)
               'pond_storage': 0.0 # pond storage
                }
            
      
        photopara = {
                  'All data':
                      {'beta':0.513,
                       'gamma':0.0196,
                       'kappa': -0.389,
                       'tau':7.2,
                       'X0': -4.0,
                       'Smax':17.3,
                       'alfa': 1.,
                       'nu': 5.
                       },
                  'Sodankyla':
                      {'beta':0.831,
                       'gamma':0.065,
                       'kappa': -0.150,
                       'tau':10.2,
                       'X0': -0.9,
                       'Smax':16.4,
                       'alfa': 1.,
                       'nu': 5.
                       },
                  'Hyytiala':
                      {'beta':0.504,
                       'gamma':0.0303,
                       'kappa': -0.235,
                       'tau':11.1,
                       'X0': -3.1,
                       'Smax':17.3,
                       'alfa': 1.,
                       'nu': 5.
                       },
                  'Norunda':
                      {'beta':0.500,
                       'gamma':0.0220,
                       'kappa': -0.391,
                       'tau':5.7,
                       'X0': -4.0,
                       'Smax':17.6,
                       'alfa':1.062,
                       'nu': 11.27,
                       },
                  'Tharandt':
                      {'beta':0.742,
                       'gamma':0.0267,
                       'kappa': -0.512,
                       'tau':1.8,
                       'X0': -5.2,
                       'Smax':18.5,
                       'alfa':1.002,
                       'nu': 442.
                       },
                  'Bray':
                      {'beta':0.459,
                       'gamma':-0.000669,
                       'kappa': -0.560,
                       'tau':2.6,
                       'X0': -17.6,
                       'Smax':45.0,
                       'alfa':0.843,
                       'nu': 2.756,
                       },
                               }
        #----------- Arrange and make coherent------
        #cpara['lat']=wpara[wlocation]['lat']; cpara['lon']=wpara[wlocation]['lon']
    
        o_w = wpara[wlocation] if wlocation is not None  else wpara 
        o_p = photopara[photosite] if photosite is not None else photopara  
        return o_w, cpara, org_para, o_p
    
    def print_team(self):
        inp= (u"Suosimulaattori Susi on kehitetty Itä-Suomen yliopiston, Luken, Helsingin yliopiston ja Suomen metsäkeskuksen yhteistyönä.  " 
              "\n"
              "\n"
              "-------- KEHITYSTIIMI ---------"
              "\n"
              u"- Ari Laurén, Itä-Suomen yliopisto "
              "\n"
              "---------Metsätieteet-----------"
              "\n"              
              u"- Hannu Hökkä & Leena Stenberg, Luke" 
              "\n"
              u" - Ari Laurén, Itä-Suomen yliopisto" 
              "\n"
              "---------Hydrologia----------------"
              "\n"
              u"- Samuli Launiainen & Kersti Haahti, Luke " 
              "\n"
              u" - Ari Laurén, Itä-Suomen yliopisto"
              "\n"
              "---------Biogeokemia----------"
              "\n"
              u" - Marjo Palviainen, Helsingin yliopisto"
              "\n"              
              u"- Raija Laiho, Luke" 
              "\n"
              "-----------------------------------"
              "\n"              
              "Yhteydenotot:" 
              "\n"
              u"Apulaisprofessori Ari Laurén,"
              "\n"
              u"Itä-Suomen yliopisto, Luonnontieteiden ja metsätieteiden tiedekunta, Joensuun kampus"
              "\n"
              u"ari.lauren@uef.fi"
              )
        self.textEditTeam.append(inp)

    def get_ini_para(self, folderName=None, hdomSim=None, volSim=None, scen = None,
                      ageSim=None, sarkaSim=None, susiPath = None, ddwest=None, ddeast=None, n=None,
                      peatN=None, peatP=None, peatK=None):
    
        #********** Stand parameters and weather forcing*******************
        #--------------Weather variables 10 km x 10 km grid   
        if susiPath is None: susiPath=""
        # Hannun parametrit
        #------------ Soil and stand parameters ----------------------------------
        #bd = {'sfc_2': 0.14, 'sfc_3': 0.11, 'sfc_4': 0.10, 'sfc_5': 0.08}           # Mese study: bulk densities in different fertility classes                                                                 # peat layer thickness, cm            

        spara ={ 
    
            'Ruohoturvekangas, rahkaturve':{
            'species': 'Pine', 'sfc':2, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.14, 0.14, 0.14, 0.14, 0.14, 0.14], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK
             },  

            'Ruohoturvekangas, muu turve':{
            'species': 'Pine', 'sfc':2, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.14, 0.14, 0.14, 0.14, 0.14, 0.14], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK,
            'depoN': 4.0, 'depoP':0.1, 'depoK':1.0

             },  

            'Mustikkaturvekangas, rahkaturve':{
            'species': 'Pine', 'sfc':3, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.11, 0.11, 0.11, 0.11, 0.11, 0.11], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK,
            'depoN': 4.0, 'depoP':0.1, 'depoK':1.0

             },  

            'Mustikkaturvekangas, muu turve':{
            'species': 'Pine', 'sfc':3, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.11, 0.11, 0.11, 0.11, 0.11, 0.11], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK,
            'depoN': 4.0, 'depoP':0.1, 'depoK':1.0

             },  

            'Puolukkaturvekangas, rahkaturve':{
            'species': 'Pine', 'sfc':4, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.10, 0.10, 0.10, 0.10, 0.10, 0.10], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK,
            'depoN': 4.0, 'depoP':0.1, 'depoK':1.0

             },  
                    
            'Puolukkaturvekangas, muu turve':{
            'species': 'Pine', 'sfc':4, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.10, 0.10, 0.10, 0.10, 0.10, 0.10], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK,
            'depoN': 4.0, 'depoP':0.1, 'depoK':1.0

 
            },  

            'Varputurvekangas, rahkaturve':{
            'species': 'Pine', 'sfc':5, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.08, 0.08, 0.08, 0.08, 0.08, 0.08], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK,
            'depoN': 4.0, 'depoP':0.1, 'depoK':1.0

             },  
                    
            'Varputurvekangas, muu turve':{
            'species': 'Pine', 'sfc':5, 'hdom':hdomSim, 'vol':volSim, 'age':ageSim, 'smc': 'Peatland',
            'nLyrs':30, 'dzLyr': 0.05, 'L': sarkaSim, 'n':n, 
            'ditch depth west': ddwest,   #nLyrs kerrosten lkm, dzLyr kerroksen paksuus m, saran levys m, n laskentasolmulen lukumäärä, ditch depth pjan syvyys simuloinnin alussa m  
            'ditch depth east': ddeast,
            'ditch depth 20y west': ddwest,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'ditch depth 20y east': ddeast,                                            #ojan syvyys 20 vuotta simuloinnin aloituksesta
            'scenario name': scen, #kasvunlisaykset
            'initial h': -0.2, 'slope': 0.0, 
            'peat type':['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A'], 
            'peat type bottom':['A'],'anisotropy':10.,
            'vonP': True,
            'vonP top':  [3,3,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5], 
            'vonP bottom': 8,
            'bd top':[0.08, 0.08, 0.08, 0.08, 0.08, 0.08], 'bd bottom': 0.16,
            'peatN':peatN, 'peatP':peatP, 'peatK':peatK,
            'depoN': 4.0, 'depoP':0.1, 'depoK':1.0

             },  
    
                }
        #------------  Output parameters -------------------------------------------------    
        outpara ={
            'outfolder':folderName, 
            #'outfolder': newfolder(folderName),
            'ofile': 'out2.xls', 'tsfile': 'ts', 'gwl_file': 'gwl', 'gr_file': 'gr.xls', 'runfile': 'roff',
            'startday': 1, 'startmonth':7, # Päivä, josta keskiarvojen laskenta alkaa
            'endday':31, 'endmonth':8, # Päivä, johon keskiarvojen laskenta loppuu
            'figs': True, 'to_file':False, 'static stand':False, 'hydfig':False,
            }    
    
        #o_s = spara[sitetype] if sitetype is not None else spara
        return  spara, outpara



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindowSusi = QtWidgets.QMainWindow()
    ui = Ui_MainWindowSusi()
    ui.setupUi(MainWindowSusi)
    MainWindowSusi.show()
    sys.exit(app.exec_())

