"""Code for bagPakdGui.py: Launches the GUI of the BagPakd program.

Inputs Arguments: None
Output Aruments : None

Description: Standalone GUI, built using pyQT5. Helps user select images for
             processing. After images are selected, camera/image parameters 
             are entered by the user. Clicking the calculate button returns
             dimension and classification results. 

Help: See status bar in the GUI for directions

Guide:
    1. Load folder with images
    2. Select Images
    3. Enter Camera Parameters (order does not matter for steps 2 and 3)
    4. Rotate images if necessary
    5. Click Calculate for results              

User defined modules imported:
    1. bagDimensionModule, [Author: Nick Theis]
    2. bagClassificationModule, [Author: Erik Gough]
"""

__author__  = 'R Vas'
_credits__  = ['Nick Theis', 'Erik Gough']
__status__  = 'Dev'
__version__ = 0.9
__date__    = 10292018


import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# import dimension code as bagDim
from bag_detection import bagDimV3
# import classification code as bagClassify
from bag_classifier import bagClassify
# parallel processing libraries
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
import gdriveSync as gd

class StartWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()   
        
        # Script Directory
        self.scriptDir = os.path.dirname(os.path.realpath(__file__))
        
        # Initializing GUI
        self.initUI()  
        
        # Initializing Input Parameters
        self.camData = {'deltaImageDist':0,
                   'camFocalLength':0,
                   'camVerticalPixelCount':0,
                   'camHorizontalPixelCount':0,
                   'camPixelSize':0}
        
        # Initializing Output
        self.bagDimensions = {'Height':999,
                   'Length':999,
                   'Width':999,}
        
        self.bagClass = {'Classification': 'Model Connection in Work'}        
        
    def initUI(self):
        
        # Initialize status bar
        self.statusBar().showMessage('Ready - Start with Load Folder')
        
        #Initialize toolbar
        self.tb = self.addToolBar('ToolBar')
        self.tb.setStatusTip('Toolbar Options')
        
        # Define Action to load folder -connect function getFolderName
        self.actionLf = QAction('Load Folder')
        self.actionLf.setStatusTip('Choose folder with images')
        self.actionLf.triggered.connect(self.getFolderName)
        self.tb.addAction(self.actionLf)
        
        # Define Action to set Pitch mode for dimension calculator
        self.actionPm = QAction('Pitch Mode')
        self.actionPm.setStatusTip('Only image 1 Used and Distance to Object')
        self.actionPm.setCheckable(True)
        self.actionPm.setChecked(True)
        self.actionPm.triggered.connect(self.pmToggle)
        self.tb.addAction(self.actionPm)
        self.mode = 1        
        
        # Define Action to set Dual Image mode for dimension calculator
        self.actionDm = QAction('Dual Image Mode')
        self.actionDm.setStatusTip('Image 1 and 2 Used and Distance between images')
        self.actionDm.setCheckable(True)
        self.actionDm.setChecked(False)
        self.actionDm.triggered.connect(self.dmToggle)
        self.tb.addAction(self.actionDm)
        
        # Define Action to open MySQL - connect function __________
        self.actionMySQL = QAction('MySQL Database')
        self.actionMySQL.setStatusTip('Open MqSQL Results Database')
        self.tb.addAction(self.actionMySQL)   
        
        # Define Action to Rotate Image 1 by 90 deg
        self.actionR1 = QAction('Rotate Image 1')
        self.actionR1.setStatusTip('Rotate Image 1 by 90 Degrees')
        self.actionR1.triggered.connect(lambda: self.rotateImage(1))
        self.tb.addAction(self.actionR1)
        
        # Define Action to Rotate Image 2 by 90 deg
        self.actionR2 = QAction('Rotate Image 2')
        self.actionR2.setStatusTip('Rotate Image 2 by 90 Degrees')
        self.actionR2.triggered.connect(lambda: self.rotateImage(2))
        self.tb.addAction(self.actionR2)
        
        # Define Action to sync Google Drive
        self.actionSync = QAction('Sync')
        self.actionSync.setStatusTip('Sync Google Drive')
        self.actionSync.triggered.connect(self.sync)
        self.tb.addAction(self.actionSync)
        
        # Set local folder path where google drive contents are downloaded
        self.gdrive = os.path.join(os.getcwd(),'GoogleDriveContents')
        
        # Create file explorer model
        self.fileExplorerModel = QFileSystemModel()
        self.fileExplorerModel.setRootPath(QDir.rootPath())
        
        # Create a treeview and set file explorer model to the treeview
        self.fileViewer = QTreeView(self)
        self.fileViewer.setModel(self.fileExplorerModel)
        self.fileViewer.clicked.connect(self.openImage)
        self.fileViewer.setMinimumSize(420,490)
        self.fileViewer.move(10,80)
        self.fileViewer.setRootIndex(self.fileExplorerModel.index(self.gdrive))
        self.fileViewer.hideColumn(1)
        self.fileViewer.show()
        
        # Define label for image 1 (lb) and filename label (lf) 
        self.lf1=QLabel(self)
        self.lf1.move(450,20)
        self.lf1.setMinimumWidth(400)
        self.lb1=QLabel(self)
        self.lb1.move(450,50)
        self.lb1.setStatusTip('Image 1')
        
        # Define label for image 2 (lb) and filename label (lf) 
        self.lf2=QLabel(self)
        self.lf2.move(860,20)
        self.lf2.setMinimumWidth(400)
        self.lb2=QLabel(self)
        self.lb2.move(860,50)
        self.lb2.setStatusTip('Image 2')
        
        self.imgPos = 1
        self.imageCount = 0
        
        # Define frame of camera variables
        self.camFrame = QFrame(self)
        self.camFrame.resize(420,200)
        self.camFrame.move(10,580)
        self.camFrame.setFrameShadow(QFrame.Raised)
        self.camFrame.setFrameShape(QFrame.WinPanel)
        self.camFrame.setLineWidth(3)
        self.camFrame.setStyleSheet("background-color: rgb(255,230,234)")
        self.camFrame.setStatusTip('Camera/Image Inputs')
        self.camFrame.hide()
        
        # Define camera variables label and input field for var 1
        self.camLb1=QLabel(self)
        self.camLb1.setText('Object Distance [ft]')
        self.camLb1.move(50,600)
        self.camLb1.setMinimumWidth(200)
        self.q1=QLineEdit(self)
        self.q1.move(250,600)
        self.q1.resize(50,20)
        self.q1.setStatusTip('Distance')
        self.camLb1.hide()
        self.q1.hide()
        self.q1.textChanged.connect(self.chkCamInput)
        
        # Define camera variables label and input field for var 2
        self.camLb2=QLabel(self)
        self.camLb2.setText('Camera Focal Length [m]')
        self.camLb2.move(50,630)
        self.camLb2.setMinimumWidth(200)
        self.q2=QLineEdit(self)
        self.q2.move(250,630)
        self.q2.resize(50,20)
        self.q2.setStatusTip('Enter camera focal length in meters')
        self.camLb2.hide()
        self.q2.hide()
        self.q2.textChanged.connect(self.chkCamInput)
        
        # Define camera variables label and input field for var 3
        self.camLb3=QLabel(self)
        self.camLb3.setText('Camera Vertical Pixel Count')
        self.camLb3.move(50,660)
        self.camLb3.setMinimumWidth(200)
        self.q3=QLineEdit(self)
        self.q3.move(250,660)
        self.q3.resize(50,20)
        self.q3.setStatusTip('Enter camera vertical pixel count')
        self.camLb3.hide()
        self.q3.hide()
        self.q3.textChanged.connect(self.chkCamInput)
        
        # Define camera variables label and input field for var 4
        self.camLb4=QLabel(self)
        self.camLb4.setText('Camera Horizontal Pixel Count')
        self.camLb4.move(50,690)
        self.camLb4.setMinimumWidth(200)
        self.q4=QLineEdit(self)
        self.q4.move(250,690)
        self.q4.resize(50,20)
        self.q4.setStatusTip('Enter camera horizontal pixel count')
        self.camLb4.hide()
        self.q4.hide()
        self.q4.textChanged.connect(self.chkCamInput)
        
        # Define camera variables label and input field for var 5
        self.camLb5=QLabel(self)
        self.camLb5.setText('Camera Pixel Size [m]')
        self.camLb5.move(50,720)
        self.camLb5.setMinimumWidth(200)
        self.q5=QLineEdit(self)
        self.q5.move(250,720)
        self.q5.resize(50,20)
        self.q5.setStatusTip('Enter camera pixel size in meters')
        self.camLb5.hide()
        self.q5.hide()
        self.q5.textChanged.connect(self.chkCamInput)
        
        self.camLb1.show()
        self.q1.show()
        self.camLb2.show()
        self.q2.show()
        self.camLb3.show()
        self.q3.show()
        self.camLb4.show()
        self.q4.show()
        self.camLb5.show()
        self.q5.show()
        self.camFrame.show()
                
        # Define Dimension Results frame 
        self.dimFrame = QFrame(self)        
        self.dimFrame.resize(420,80)
        self.dimFrame.move(10,790)
        self.dimFrame.setFrameShadow(QFrame.Raised)
        self.dimFrame.setFrameShape(QFrame.WinPanel)
        self.dimFrame.setLineWidth(3)
        self.dimFrame.setStatusTip('Dimension Results')
        self.dimFrame.setStyleSheet("background-color: rgb(255,255,255)")
        self.dimFrame.hide() 
        
        self.grid = QGridLayout()
        self.bagHeight = QLabel('',self)
        self.bagLength = QLabel('',self)
        self.bagWidth = QLabel('',self)
        self.bagClassification = QLabel('',self)
        self.grid.addWidget(QLabel('Results'),1,1)
        self.grid.addWidget(QLabel(' '),1,2)
        self.grid.addWidget(QLabel(' '),1,3)
        self.grid.addWidget(QLabel(' '),1,4)
        self.grid.addWidget(QLabel(' '),1,5)  
        self.grid.addWidget(QLabel(' '),1,6)  
        self.grid.addWidget(QLabel('Bag Height = '),2,1)
        self.grid.addWidget(self.bagHeight,2,2)
        self.grid.addWidget(QLabel('Bag Length = '),2,3)
        self.grid.addWidget(self.bagLength,2,4)
        self.grid.addWidget(QLabel('Bag Width = '),2,5)
        self.grid.addWidget(self.bagWidth,2,6)   
        self.dimFrame.setLayout(self.grid)  
        
        # Define Classification Results frame 
        self.classFrame = QFrame(self)        
        self.classFrame.resize(420,50)
        self.classFrame.move(10,870)
        self.classFrame.setFrameShadow(QFrame.Raised)
        self.classFrame.setFrameShape(QFrame.WinPanel)
        self.classFrame.setLineWidth(3)
        self.classFrame.setStatusTip('Classification Results')
        self.classFrame.setStyleSheet("background-color: rgb(255,255,255)") 
        self.classFrame.hide() 
        
        self.grid2 = QGridLayout()
        self.grid2.addWidget(QLabel('Classification = '),1,1)
        self.grid2.addWidget(self.bagClassification,1,2)
        self.grid2.addWidget(QLabel(' '),1,3)
        self.grid2.addWidget(QLabel(' '),1,4)
        self.grid2.addWidget(QLabel(' '),1,5)
        self.grid2.addWidget(QLabel(' '),1,6)        
        self.classFrame.setLayout(self.grid2)                
        
        # Create calculate button and conenct it to 
        self.calcButton = QPushButton('Calculate Dimensions and Classify', self)
        self.calcButton.setStatusTip('Click for Results')
        self.calcButton.resize(self.calcButton.sizeHint())
        self.calcButton.move(100,40)
        self.calcButton.setStyleSheet("background-color: rgb(255,230,234)")
        self.calcButton.hide()
        self.calcButton.clicked.connect(self.runBagpakd)
        
        # Set default size of window
        self.resize(1300,950)
        self.setWindowTitle('BagPAKD: Bag Processing, Analysis, Klassification, and Dimensions [GUI] [CS501]')    
        self.setWindowIcon(QIcon(self.scriptDir + os.path.sep + 'bagIcon.png')) 
        self.show()
        
############################## End Init Method ############################### 
        
############################## Class Methods #################################
    
    def sync(self):
        gd.sync()
    
    def pmToggle(self,state):
        if state:
            self.actionPm.setChecked(True)
            self.actionDm.setChecked(False)
            self.camLb1.setText('Object Distance [ft]')
            self.mode = 1
            self.imgPos = 1
            self.lb2.clear()
        else:
            self.actionPm.setChecked(False)
            self.actionDm.setChecked(True)
            self.camLb1.setText('Image Distance Delta [m]')
            self.mode = 2  
        
    def dmToggle(self,state):
        if state:
            self.actionPm.setChecked(False)
            self.actionDm.setChecked(True)
            self.camLb1.setText('Image Distance Delta [m]')
            self.mode = 2
        else:
            self.actionPm.setChecked(True)
            self.actionDm.setChecked(False)
            self.camLb1.setText('Object Distance [ft]')
            self.mode = 1
            self.imgPos = 1
            self.lb2.clear()            
    
    def chkCamInput(self):
        if (self.q1.text() and self.q2.text() and self.q3.text() and self.q4.text() \
        and self.q5.text()):
            self.camFrame.setStyleSheet("background-color: rgb(213,255,204)")
            self.calcButton.setStyleSheet("background-color: rgb(213,255,204)")
        else:
            self.camFrame.setStyleSheet("background-color: rgb(255,230,234)") 
            self.calcButton.setStyleSheet("background-color: rgb(255,230,234)")
    
    # Define slot function getFolderName 
    def getFolderName(self):
        self.folderName = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.fileViewer.setRootIndex(self.fileExplorerModel.index(self.folderName))
        self.fileViewer.hideColumn(1)
        self.fileViewer.show()
        self.camLb1.show()
        self.q1.show()
        self.camLb2.show()
        self.q2.show()
        self.camLb3.show()
        self.q3.show()
        self.camLb4.show()
        self.q4.show()
        self.camLb5.show()
        self.q5.show()
        self.camFrame.show()
   
    def getSensorFile(self,imageName):
        files = [f for f in os.listdir(self.gdrive) if os.path.isfile(os.path.join(self.gdrive,f))]
        imgSplit = imageName.split('_')
        
        for f in files:
            sensorSplit = f.split('_')
            if 'Sensor' in sensorSplit[0]:
                if imgSplit[1] == sensorSplit[1]:
                    if imgSplit[-1][0:5] == sensorSplit[-1][0:5]:
                        sensorFileName = f
                        break;
        return sensorFileName
    
    def getObjectDistance(self,sensorFileName):
        f =  open(os.path.join(self.gdrive,sensorFileName),'r')
        data = f.read() 
        f.close()
        objectDistance = data.split(',')[-1] 
        return objectDistance
                
    # Define slot function openImage 
    def openImage(self,index):
        
        if self.mode == 1:
            imageName = self.fileExplorerModel.fileName(index)        
            try:
                sensorFileName = self.getSensorFile(imageName)
            except:
                pass
            else:
                objectDistance = self.getObjectDistance(sensorFileName)
                objectDistance = round(float(objectDistance),4) 
                self.q1.setText(str(objectDistance))
        else:
            self.q1.setText(str('0.2'))

        if self.imgPos==1:
            self.imageFile = self.fileExplorerModel.fileInfo(index).absoluteFilePath()
            self.lf1.setText(self.imageFile);
            self.pixmap1 = QPixmap(self.imageFile)            
            self.pixmap1 = self.pixmap1.scaled(400,400,Qt.KeepAspectRatio)            
            self.lb1.resize(self.pixmap1.width(),self.pixmap1.height())
            self.lb1.setPixmap(self.pixmap1)
            if self.mode==2:
                self.imgPos=2
            else:
                self.imgPos=1
            self.imageCount = self.imageCount + 1
        elif self.imgPos==2 and self.mode==2:
            self.imageFile = self.fileExplorerModel.fileInfo(index).absoluteFilePath()
            self.lf2.setText(self.imageFile);
            self.pixmap2 = QPixmap(self.imageFile)
            self.pixmap2 = self.pixmap2.scaled(400,400,Qt.KeepAspectRatio)
            self.lb2.resize(self.pixmap2.width(),self.pixmap2.height())
            self.lb2.setPixmap(self.pixmap2)
            self.imgPos=1
            self.imageCount = self.imageCount + 1
        
        if self.mode==2:
            if self.imageCount >1:
                self.calcButton.show()
        else:
            if self.imageCount >0:
                self.calcButton.show()
     
    # Define function to rotate image
    def rotateImage(self,n):
        if n==1:
            tr = QTransform().rotate(90)
            self.pixmap1 = self.pixmap1.transformed(tr,Qt.SmoothTransformation)
            self.pixmap1 = self.pixmap1.scaled(400,400,Qt.KeepAspectRatio)            
            self.lb1.resize(self.pixmap1.width(),self.pixmap1.height())
            self.lb1.setPixmap(self.pixmap1)
        elif n==2:
            tr = QTransform().rotate(90)
            self.pixmap2 = self.pixmap2.transformed(tr,Qt.SmoothTransformation)
            self.pixmap2 = self.pixmap2.scaled(400,400,Qt.KeepAspectRatio)            
            self.lb2.resize(self.pixmap2.width(),self.pixmap2.height())
            self.lb2.setPixmap(self.pixmap2)
            
    # Function that is run when the calculate button is clicked    
    def runBagpakd(self):
        try:
            if self.mode == 1:
                self.camData['objectDist']=float(self.q1.text())
            else:
                self.camData['deltaImageDist']=float(self.q1.text())
            self.camData['camFocalLength']=float(self.q2.text())
            self.camData['camVerticalPixelCount']=float(self.q3.text())
            self.camData['camHorizontalPixelCount']=float(self.q4.text())
            self.camData['camPixelSize']=float(self.q5.text())
        except ValueError:
            self.runErr = -1
            self.statusBar().showMessage('Error: Enter Values for Camera/Image Variables')
        else:    
            print(self.camData)
             
            # Nick pass self.camData to your function and return bag 
            # Dimension module results:
            # self.mode = 1 for pitch mode
            # self.mode = 2 for dual image mode
            self.bagDimensions = bagDimV3.run(self.camData, self.lf1.text(),self.lf2.text(),self.mode)
            # example to output to GUI (assign results here):            
            self.bagHeight.setText(str(self.bagDimensions['Height']))
            self.bagLength.setText(str(self.bagDimensions['Length']))
            self.bagWidth.setText(str(self.bagDimensions['Width']))
            
            print(self.bagDimensions)
             
            # Set up the paths to the models and labels
            models = ["bag_inceptionv3.model", "bagclassifier.model", "bag_xception.model"]
            path = "./bag_classifier"
            labels = "labels.bin"
            print(self.lf1.text())
            
            # run the 3 classifications in parallel to reduce processing time
            classification_results = Parallel(n_jobs=3)(delayed(bagClassify.classifyImage)(path, self.lf1.text(), m, labels) for m in models)
            probabilities = [r[1] for r in classification_results]
            classifications = [r[0] for r in classification_results]
            
            # find the classification that occurs the most out of
            # the three models, or return the single classification
            # with the highest probability
            probabilities_tup, classifications_tup = zip(*sorted(zip(probabilities, classifications), reverse=True))
            classification,num_classification = Counter(classifications_tup).most_common(1)[0]
            print(classification,num_classification)

            # assign the final classification
            self.bagClass = classification
            print(self.bagClass)

            # update the GUI text
            self.bagClassification.setText(self.bagClass + " " + str(round(probabilities_tup[0],1)) + "%")
            
            # Show result frame
            self.dimFrame.show()
            self.classFrame.show()            
            
############################ End Class ########################################        
# Run GUI        
if __name__ == '__main__':    
    app = QApplication(sys.argv)
    s = StartWindow()
    sys.exit(app.exec_())
