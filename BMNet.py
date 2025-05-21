import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pandas as pd
import os
import numpy as np

# 显存配置
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 使用CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 配置参数
CSV_PATH = 'b m data/labels.csv'
IMG_DIR = 'b m data/images'
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 30

def create_data_generators():
    """创建训练集和验证集数据生成器"""
    # 读取数据并转换标签为字符串
    df = pd.read_csv(CSV_PATH, dtype={'b_m': str})
    
    # 路径处理（处理路径中的空格）
    df['image_path'] = df['image'].apply(
        lambda x: os.path.normpath(os.path.join(IMG_DIR, x))
    )
    
    # 验证文件存在性
    missing_files = df[~df['image_path'].apply(os.path.exists)]
    if not missing_files.empty:
        print(f"警告：发现{len(missing_files)}个缺失文件，已自动过滤")
        df = df[df['image_path'].apply(os.path.exists)]
    
    # 检查标签是否为二分类（"0"/"1"）
    unique_labels = df['b_m'].unique()
    if not set(unique_labels).issubset({"0", "1"}):
        raise ValueError("标签必须为'0'或'1'的字符串格式")
    
    # 数据增强配置
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2  # 20%作为验证集
    )
    
    # 训练集生成器（二分类使用binary模式）
    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='b_m',
        target_size=IMG_SIZE,
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='training',
        shuffle=True,
        classes=["0", "1"]
    )
    
    # 验证集生成器
    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='b_m',
        target_size=IMG_SIZE,
        class_mode='binary',
        batch_size=BATCH_SIZE,
        subset='validation',
        shuffle=False,
        classes=["0", "1"]
    )
    
    # 打印数据分布
    print("\n训练集样本数:", train_gen.samples)
    print("验证集样本数:", val_gen.samples)
    print("类别分布 - 训练集:", 
          {k: v for k, v in zip(['0', '1'], np.bincount(train_gen.labels))})
    print("类别分布 - 验证集:", 
          {k: v for k, v in zip(['0', '1'], np.bincount(val_gen.labels))})
    
    return train_gen, val_gen

def build_binary_model():
    """构建二分类模型"""
    base_model = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    # 冻结预训练层
    base_model.trainable = False
    
    # 自定义顶层
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    model.summary()
    return model

def train_model():
    """训练流程"""
    try:
        # 创建数据生成器
        train_gen, val_gen = create_data_generators()
        
        # 构建模型
        model = build_binary_model()
        
        # 回调函数配置
        callbacks = [
            ModelCheckpoint(
                'best_bm_model.h5',
                monitor='val_auc',  # 使用验证集AUC作为监控指标
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',  # 使用验证集loss
                factor=0.5,
                patience=3,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True
            )
        ]
        
        # 执行训练
        history = model.fit(
            train_gen,
            validation_data=val_gen,  # 添加验证集
            epochs=EPOCHS,
            callbacks=callbacks,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen)
        )
        
        # 保存最终模型
        model.save('BMNet_final.h5')
        print("\n模型训练完成!")
        
        # 评估结果
        print("\n训练结果:")
        print(f"验证集准确率: {max(history.history['val_accuracy']):.2%}")
        print(f"验证集AUC: {max(history.history['val_auc']):.2%}")
        
    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        print("\n排查建议:")
        print("1. 检查CSV文件格式是否正确")
        print("2. 确认'b_m'列的值是字符串格式的'0'和'1'")
        print("3. 验证所有图像文件均为有效格式")
        print("4. 确保路径中的空格已正确处理")

if __name__ == "__main__":
    train_model()