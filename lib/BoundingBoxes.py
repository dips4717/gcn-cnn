from lib import BoundingBox
from utils import *


class BoundingBoxes:
    def __init__(self):
        self.boundingBoxes = []

    def addBoundingBox(self, bb):
        self.boundingBoxes.append(bb)

    def removeBoundingBox(self, _boundingBox):
        for d in self.boundingBoxes:
            if BoundingBox.compare(d, _boundingBox):
                del self.boundingBoxes[d]
                return

    def removeAllBoundingBoxes(self):
        self.boundingBoxes = []

    def getBoundingBoxes(self):
        return self.boundingBoxes

    def getBoundingBoxByClass(self,classId ):
        boundingBoxes = []
        for d in self.boundingBoxes:
            if d.getClassId() == classId:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self.boundingBoxes if d.getImageName() == imageName]

    def count(self, bbType=None):
        count = 0
        for d in self.boundingBoxes:
            count += 1
        return count

    def getClasses(self):
        classes = []
        for d in self.boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getIconClasses(self):
        iconClasses = []
        for d in self.boundingBoxes:
            c = d.getItemClass()
            if c not in iconClasses:
                iconClasses.append(c)
        return iconClasses
    
    
    def getTextButtonClasses(self):
        textButtonClasses = []
        for d in self.boundingBoxes:
            c = d.getTextButtonClass()
            if c not in textButtonClasses:
                textButtonClasses.append(c)
        return textButtonClasses




#    def clone(self):
#        newBoundingBoxes = BoundingBoxes()
#        for d in self._boundingBoxes:
#            det = BoundingBox.clone(d)
#            newBoundingBoxes.addBoundingBox(det)
#        return newBoundingBoxes

#    def drawAllBoundingBoxes(self, image, imageName):
#        bbxes = self.getBoundingBoxesByImageName(imageName)
#        for bb in bbxes:
#            if bb.getBBType() == BBType.GroundTruth:  # if ground truth
#                image = add_bb_into_image(image, bb, color=(0, 255, 0))  # green
#            else:  # if detection
#                image = add_bb_into_image(image, bb, color=(255, 0, 0))  # red
#        return image

    # def drawAllBoundingBoxes(self, image):
    #     for gt in self.getBoundingBoxesByType(BBType.GroundTruth):
    #         image = add_bb_into_image(image, gt ,color=(0,255,0))
    #     for det in self.getBoundingBoxesByType(BBType.Detected):
    #         image = add_bb_into_image(image, det ,color=(255,0,0))
    #     return image
