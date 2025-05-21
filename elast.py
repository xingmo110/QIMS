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

CSV_PATH = 'elast_data/labels.csv'
IMG_DIR = 'elast_data/images'
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 5

def create_data_flow():
    df = pd.read_csv(CSV_PATH, dtype={'RTE': str})
    
    df['image_path'] = df['image'].apply(
        lambda x: os.path.normpath(os.path.join(IMG_DIR, x))
    )
    
    valid_labels = {'0', '1', '2', '3', '4'}
    invalid_rows = df[~df['RTE'].isin(valid_labels)]
    
    if not invalid_rows.empty:
        print(f"Found {len(invalid_rows)} invalid labels, filtered")
        print("Invalid samples:")
        print(invalid_rows.head())
        df = df[df['RTE'].isin(valid_labels)]
    
    missing_files = df[~df['image_path'].apply(os.path.exists)]
    if not missing_files.empty:
        print(f"Found {len(missing_files)} missing files, filtered")
        df = df[df['image_path'].apply(os.path.exists)]
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='RTE',
        target_size=IMG_SIZE,
        class_mode='sparse',
        batch_size=BATCH_SIZE,
        subset='training',
        classes=[str(i) for i in range(5)]
    )
    
    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='RTE',
        target_size=IMG_SIZE,
        class_mode='sparse',
        batch_size=BATCH_SIZE,
        subset='validation',
        classes=[str(i) for i in range(5)]
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    return train_gen, val_gen

def build_multiclass_model():
    base = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    
    output = Dense(NUM_CLASSES, activation='softmax', name='RTE')(base.output)
    
    model = Model(inputs=base.input, outputs=output)
    
    optimizer = Adam(
        learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    try:
        train_gen, val_gen = create_data_flow()
        
        model = build_multiclass_model()
        model.summary()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen)
        )
        
        model.save('elast_model_final.h5')
        print("Model training completed and saved")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Troubleshooting suggestions:")
        print("1. Verify RTE column values are strings from '0' to '4'")
        print("2. Check for spaces in image paths (recommend renaming folder to elast_data)")
        print("3. Ensure all images are valid JPEG/PNG format")