# --- Training Phase ---
import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import sounddevice as sd
from xarm.wrapper import XArmAPI
import time

# Parameters
DATA_PATH = r'C:\Users\Admin\Desktop\data'  # Path to folders: data/one, data/two, data/three
SAMPLE_RATE = 16000
TARGET_SHAPE = (64, 64)
CLASSES = ['one', 'two', 'three']

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=TARGET_SHAPE[1], axis=1)
    mfcc = np.resize(mfcc, TARGET_SHAPE)
    mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
    return mfcc

def load_dataset():
    X, y = [], []
    for idx, label in enumerate(CLASSES):
        folder = os.path.join(DATA_PATH, label)
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                features = extract_features(os.path.join(folder, file))
                X.append(features)
                y.append(idx)
    X = np.array(X).reshape(-1, *TARGET_SHAPE, 1)
    y = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))
    return X, y

print("Loading and preprocessing dataset...")
X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")
model = models.Sequential([
    layers.Input(shape=(*TARGET_SHAPE, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(CLASSES), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("speech_model.h5")

# --- Deployment Phase: Voice-Controlled Robotic Arm ---
print("Loading trained model...")
model = tf.keras.models.load_model("speech_model.h5")

# Initialize xArm
arm = XArmAPI('192.168.1.155')
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)

# Define initial position
initial_position = {
    'x': 247, 'y': -2.9, 'z': 199.2,
    'roll': 166, 'pitch': -81.9, 'yaw': 11
}

def move_to_initial():
    arm.set_position(**initial_position, speed=24, wait=True)
    time.sleep(1)

def move_to_first_position():
    print("Moving to first position...")
    arm.set_position(x=247.8, y=-263.6, z=89.2, roll=167.3, pitch=-45.9, yaw=-4.7, speed=50, wait=True)
    arm.set_position(x=285.8, y=-260.3, z=75.3, roll=170.4, pitch=-37.6, yaw=-14.1, speed=50, wait=True)
    arm.set_position(x=314, y=-255.8, z=143, roll=159.7, pitch=-69.7, yaw=-1.6, speed=50, wait=True)
    move_to_initial()

def move_to_second_position():
    print("Moving to second position...")
    arm.set_position(x=404.7, y=-55.7, z=139.8, roll=177.8, pitch=-23.4, yaw=-2.7, speed=50, wait=True)
    arm.set_position(x=425.1, y=-60.7, z=91.7, roll=174.9, pitch=-20.9, yaw=3.5, speed=50, wait=True)
    arm.set_position(x=408.1, y=-80.8, z=106.7, roll=172.1, pitch=-55.9, yaw=10.4, speed=50, wait=True)
    move_to_initial()

def move_to_third_position():
    print("Moving to third position...")
    arm.set_position(x=265, y=64.5, z=114.9, roll=173.9, pitch=-40.4, yaw=46.2, speed=50, wait=True)
    arm.set_position(x=293.8, y=91.5, z=79.9, roll=174.1, pitch=-38.7, yaw=53.9, speed=50, wait=True)
    arm.set_position(x=285.3, y=-94.5, z=107.6, roll=178.1, pitch=-71.3, yaw=35.7, speed=50, wait=True)
    move_to_initial()

def move_to_feed_position():
    print("Moving to feed position...")
    arm.set_position(x=509.6, y=-29.7, z=267.8, roll=-164.6, pitch=-84.7, yaw=-20.2, speed=50, wait=True)
    move_to_initial()

def record_audio():
    audio = sd.rec(int(1 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def extract_mfcc_live(audio, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=TARGET_SHAPE[1], axis=1)
    mfcc = np.resize(mfcc, TARGET_SHAPE)
    return (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))

print("Starting voice-controlled robotic arm...")
move_to_initial()

try:
    while True:
        audio = record_audio()
        mfcc = extract_mfcc_live(audio, SAMPLE_RATE)
        mfcc = mfcc.reshape(1, *TARGET_SHAPE, 1)
        prediction = model.predict(mfcc)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"Heard: {CLASSES[predicted_class]} (Confidence: {confidence:.2f})")

        if confidence > 0.8:
            if predicted_class == 0:
                move_to_first_position()
                move_to_feed_position()
            elif predicted_class == 1:
                move_to_second_position()
                move_to_feed_position()
            elif predicted_class == 2:
                move_to_third_position()
                move_to_feed_position()

except KeyboardInterrupt:
    print("\nStopping program...")
    arm.disconnect()
    print("Arm disconnected.")