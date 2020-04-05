import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from Data import DataDownloader
import cv2


class Visualization:
    def __init__(self):
        (self.rawTrain, self.rawValidation,
         self.rawTest), self.metaData = DataDownloader().download()

    def displayTFDatasetImages(self, amountOfImages, typeOfData):
        assert isinstance(amountOfImages, int), TypeError
        data = Visualization().findCorrespondingDataType(typeOfData)
        fig = tfds.show_examples(
            self.metaData, data, rows=amountOfImages, cols=amountOfImages)

    def findCorrespondingDataType(self, typeOfData):
        assert isinstance(typeOfData, str), TypeError
        primitiveDatasetType = ['train', 'val', 'test']
        if(typeOfData == primitiveDatasetType[0]):
            data = self.rawTrain
        elif(typeOfData == primitiveDatasetType[1]):
            data = self.rawValidation
        elif(typeOfData == primitiveDatasetType[2]):
            data = self.rawTest
        else:
            raise ValueError(
                'arg: typeOfData is not in {}'.format(primitiveDatasetType))
        return data

    def displayImageWithItsMask(self, amountOfImages, typeOfData):
        image = Visualization().findCorrespondingDataType(typeOfData).shuffle(True)
        for data in image.take(amountOfImages):
            original = data['image'].numpy()
            imageCV = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(
                imageCV, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            Visualization.display(original, thresh, 'Original', 'Segmented')

    # Display two images
    @staticmethod
    def display(a, b, title1="Original", title2="Edited"):
        plt.subplot(121), plt.imshow(a), plt.title(title1)
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(b), plt.title(title2)
        plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    vs = Visualization()
    vs.displayImageWithItsMask(2, 'train')
    #vs.plotImages(2, vs.raw_train)
    # vs.displayImages(2,vs.raw_train)
