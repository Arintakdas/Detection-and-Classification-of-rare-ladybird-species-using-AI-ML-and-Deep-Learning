import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
train_path = r"C:\Users\VICTUS\Desktop\ladybird_split\train"
val_path = r"C:\Users\VICTUS\Desktop\ladybird_split\val"

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(train_path, target_size=(300, 300), class_mode='categorical')
val_gen = datagen.flow_from_directory(val_path, target_size=(300, 300), class_mode='categorical')

# Build model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=5)
model.save("ladybird_model.h5")