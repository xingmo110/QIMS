from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_preprocessor = ImageDataGenerator(
    rescale=1./127.5 - 1, 
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rotation_range=15
)

val_preprocessor = ImageDataGenerator(
    rescale=1./127.5 - 1  
)

CSV_COLUMNS = {
    'gray_features': ['morphology', 'margin', 'echo', 'structure', 'microcalcification'],
    'elast_path': 'elast_path',
    'gray_path': 'gray_path',
    'asteria_score': 'asteria_score'
}

IMG_SIZE = (299, 299)
CHANNELS = 3
BATCH_SIZE = 32