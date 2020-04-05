import tensorflow_datasets as tfds


class DataDownloader:
    def __init__(self, savePath='/home/misxy/MachineLearning/Projects/BeanClassifications/dataset/', datasetName='beans', typesOfDataset=['train', 'validation', 'test']):
        assert isinstance(savePath, str), TypeError
        assert isinstance(datasetName, str), TypeError
        assert isinstance(typesOfDataset, list), TypeError
        self.savePath = savePath
        self.datasetName = datasetName
        self.typesOfDataset = typesOfDataset

    def download(self):
        return DataDownloader().executeTFLoader()

    def executeTFLoader(self):
        return tfds.load(self.datasetName,
                         split=self.typesOfDataset,
                         data_dir=self.savePath,
                         shuffle_files=True,
                         with_info=True,
                         )
