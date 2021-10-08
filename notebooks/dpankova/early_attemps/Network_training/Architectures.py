from keras import layers
from keras import models

def GetArchitecture(ArchitectureNumber):

    print("Architecture: ",ArchitectureNumber)

    Architectures = [
        #
        # Architecture 0
        #
        [
            (layers.Conv2D, dict(filters=32, kernel_size=(11, 11), activation='relu', input_shape=(300, 60, 1))),
            (layers.MaxPooling2D, dict(pool_size=(2,2))),
            (layers.Conv2D, dict(filters=64, kernel_size=(7,7), activation='relu')),
            (layers.MaxPooling2D, dict(pool_size=(2,2))),
            (layers.Conv2D, dict(filters=32, kernel_size=(3,3), activation='relu')),
            (layers.MaxPooling2D, dict(pool_size=(2,2))),
            (layers.Flatten, dict()),
            (layers.Dropout, dict(rate=0.5)),
            (layers.Dense, dict(units=128, activation='relu')),
            (layers.Dense, dict(units=64, activation='relu')),
            (layers.Dense, dict(units=2, activation='softmax'))
        ],
        #
        # Architecture 1
        #
        [
            (layers.Conv2D, dict(filters=32, kernel_size=(50, 10), activation='relu', input_shape=(300, 60, 1))),
            (layers.MaxPooling2D, dict(pool_size=(2,2))),
            (layers.Conv2D, dict(filters=64, kernel_size=(25,5), activation='relu')),
            (layers.MaxPooling2D, dict(pool_size=(2,2))),
            (layers.Conv2D, dict(filters=32, kernel_size=(15,3), activation='relu')),
            (layers.MaxPooling2D, dict(pool_size=(2,2))),
            (layers.Flatten, dict()),
            (layers.Dropout, dict(rate=0.5)),
            (layers.Dense, dict(units=128, activation='relu')),
            (layers.Dense, dict(units=64, activation='relu')),
            (layers.Dense, dict(units=2, activation='softmax'))
        ]
            
    ]

    return Architectures[ArchitectureNumber]
        























