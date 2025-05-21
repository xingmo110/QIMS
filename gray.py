import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

CSV_PATH = 'gray_data/labels.csv'
IMG_DIR = 'gray_data/images'
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 50

def create_data_flow():
    df = pd.read_csv(CSV_PATH)
    
    df['image_path'] = df['image'].apply(
        lambda x: os.path.normpath(os.path.join(IMG_DIR, x))
    )
    
    for path in df['image_path'].head():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_dataframe(
        df,
        x_col='image_path',
        y_col=['Orientation', 'Margin', 'Echogenicity', 'Composition', 'microcalcification'],
        target_size=IMG_SIZE,
        class_mode='multi_output',
        batch_size=BATCH_SIZE,
        subset='training'
    )
    return train_gen

def build_optimized_model():
    base = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    
    outputs = [
        Dense(1, activation='sigmoid', name=name)(base.output)
        for name in ['Orientation', 'Margin', 'Echogenicity', 'Composition', 'microcalcification']
    ]
    
    model = Model(inputs=base.input, outputs=outputs)
    
    optimizer = Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

if __name__ == "__main__":
    try:
        train_gen = create_data_flow()
        model = build_optimized_model()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ]
        
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            steps_per_epoch=len(train_gen)
        )
        
        model.save('gray_model_final.h5')
        print("Model training completed and saved")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Troubleshooting suggestions:")
        print("1. Verify file paths are correct")
        print("2. Check image file integrity")
        print("3. Verify CUDA/cuDNN version compatibility")