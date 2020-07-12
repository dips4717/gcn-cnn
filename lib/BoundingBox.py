from utils import *


class BoundingBox:
    def __init__(self,
                 imageName,
                 componentLabel,
                 x,
                 y,
                 w,
                 h,
                 iconClass=None,
                 textButtonClass=None):
        """Constructor.
        Args:
            imageName: String representing the image name.
            classId: String value representing class id.
            x: Float value representing the X upper-left coordinate of the bounding box.
            y: Float value representing the Y upper-left coordinate of the bounding box.
            w: Float value representing the width bounding box.
            h: Float value representing the height bounding box.
            typeCoordinates: (optional) Enum (Relative or Absolute) represents if the bounding box
            coordinates (x,y,w,h) are absolute or relative to size of the image. Default:'Absolute'.
            imgSize: (optional) 2D vector (width, height)=>(int, int) represents the size of the
            image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
            bbType: (optional) Enum (Groundtruth or Detection) identifies if the bounding box
            represents a ground truth or a detection. If it is a detection, the classConfidence has
            to be informed.
            classConfidence: (optional) Float value representing the confidence of the detected
            class. If detectionType is Detection, classConfidence needs to be informed.
            format: (optional) Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the
            coordinates of the bounding boxes. BBFormat.XYWH: <left> <top> <width> <height>
            BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        self.imageName = imageName
        self.classId = componentLabel
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.iconClass = iconClass
        self.textButtonClass = textButtonClass

    def getBoundingBox(self):
        return (self.x, self.y, self.w, self.h)

    def getImageName(self):
        return self.imageName
    
    def getClassId(self):
        return self.classId

    def getIconClass(self):
        return self.iconClass
    
    def getTextButtonClass(self):
        return self.textButtonClass
 

    @staticmethod
    def compare(det1, det2):
        det1BB = det1.getBoundingBox
        det2BB = det2.getBoundingBox()

        if det1.getClassId() == det2.getClassId() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3]:
            return True
        return False

#    @staticmethod
#    def clone(boundingBox):
#        absBB = boundingBox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
#        # return (self._x,self._y,self._x2,self._y2)
#        newBoundingBox = BoundingBox(
#            boundingBox.getImageName(),
#            boundingBox.getClassId(),
#            absBB[0],
#            absBB[1],
#            absBB[2],
#            absBB[3],
#            typeCoordinates=boundingBox.getCoordinatesType(),
#            imgSize=boundingBox.getImageSize(),
#            bbType=boundingBox.getBBType(),
#            classConfidence=boundingBox.getConfidence(),
#            format=BBFormat.XYWH)
#        return newBoundingBox
