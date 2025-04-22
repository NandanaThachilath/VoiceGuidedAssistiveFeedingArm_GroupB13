import pybullet as p
import pybullet_data
import numpy as np
import time
import threading
import queue
import sounddevice as sd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
import librosa
import tensorflow as tf
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ====================== Voice Model Training Section ======================
data_dir = r'D:/SEM 4/Robotics/data/data'
classes = ['one', 'two', 'three']
target_shape = (64, 64)
SAMPLE_RATE = 16000

def pitch_shift(audio, sample_rate, semitones=1):
    shift = random.uniform(-semitones, semitones)
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=shift)

def extract_mfcc(file_path, sample_rate):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    if random.random() < 0.5:
        audio = pitch_shift(audio, sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc = librosa.util.fix_length(mfcc, size=target_shape[1], axis=1)
    mfcc = np.resize(mfcc, target_shape)
    return (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc) + 1e-8)

file_paths = []
labels = []
for class_name in classes:
    class_dir = os.path.join(data_dir, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith('.wav'):
            file_paths.append(os.path.join(class_dir, file_name))
            labels.append(class_name)

X_train, X_val, y_train, y_val = train_test_split(
    file_paths, labels, test_size=0.3, stratify=labels, random_state=42)

print("Processing training data...")
X_train = np.array([extract_mfcc(path, SAMPLE_RATE) for path in X_train])
print("Processing validation data...")
X_val = np.array([extract_mfcc(path, SAMPLE_RATE) for path in X_val])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

print("Training model...")
early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          batch_size=32,
          callbacks=[early_stop])

# ====================== PyBullet Setup Section ======================
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
robot_id = p.loadURDF("C:/Users/M.SRAVANTHI SUMA/feeding-assistant-robot-v1/simulationOfTheRobotArm/filesForSimulation/urdf/MATLAB_feeding_robot_arm20.urdf", useFixedBase=True)
numJoints = p.getNumJoints(robot_id)
eeLinkIndex = numJoints - 1
ts = 0.05
ee_offset_x = 0.1433

# ====================== Corrected Jacobian IK Implementation ======================
class JacobianIK:
    def init(self, robot_id, ee_link_index, h=1e-4, max_iter=200, tol=1e-3, damping=0.01):
        self.robot_id = robot_id
        self.ee_link_index = ee_link_index
        self.h = h
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.num_joints = p.getNumJoints(robot_id)
        
        # Retrieve joint limits from URDF
        self.joint_limits = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            self.joint_limits.append((joint_info[8], joint_info[9]))
        
    def compute_position(self, joint_angles):
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, joint_angles[i])
        state = p.getLinkState(self.robot_id, self.ee_link_index, computeForwardKinematics=True)
        return np.array(state[0])
    
    def compute_jacobian(self, joint_angles):
        J = np.zeros((3, self.num_joints))
        current_pos = self.compute_position(joint_angles)
        
        for k in range(self.num_joints):
            perturbed_angles = joint_angles.copy()
            perturbed_angles[k] += self.h
            new_pos = self.compute_position(perturbed_angles)
            J[:, k] = (new_pos - current_pos) / self.h
            
        return J
    
    def solve(self, target_pos, initial_angles=None):
        if initial_angles is None:
            joint_angles = np.array([p.getJointState(self.robot_id, i)[0] for i in range(self.num_joints)], dtype=np.float64)
        else:
            joint_angles = np.array(initial_angles, dtype=np.float64)
        
        for _ in range(self.max_iter):
            current_pos = self.compute_position(joint_angles)
            pos_error = target_pos - current_pos
            
            if np.linalg.norm(pos_error) < self.tol:
                break
                
            J = self.compute_jacobian(joint_angles)
            J_T = J.T
            inv_term = J @ J_T + self.damping**2 * np.eye(3)
            delta_theta = J_T @ np.linalg.pinv(inv_term) @ pos_error
            
            # Apply joint limits during update
            joint_angles += delta_theta
            for i in range(self.num_joints):
                joint_angles[i] = np.clip(joint_angles[i], 
                                        self.joint_limits[i][0], 
                                        self.joint_limits[i][1])
            
        return joint_angles

# ====================== Trajectory Generation ======================
def generate_trajectory(waypoints, orientations, total_time):
    num_points = int(total_time / ts) + 1
    time_stamps = np.linspace(0, total_time, len(waypoints))
    
    pos_interp = interp1d(time_stamps, waypoints, axis=0, kind='cubic')
    pos_traj = pos_interp(np.linspace(0, total_time, num_points))
    
    return pos_traj, None  # Orientation not used

def calculate_ik_trajectory(pos_traj):
    joint_traj = np.zeros((numJoints, pos_traj.shape[0]), dtype=np.float64)
    ik_solver = JacobianIK(robot_id, eeLinkIndex)
    
    # Initialize with current joint positions within limits
    ikInitGuess = []
    for i in range(numJoints):
        joint_info = p.getJointInfo(robot_id, i)
        ikInitGuess.append(np.clip(p.getJointState(robot_id, i)[0],
                                 joint_info[8],
                                 joint_info[9]))
    ikInitGuess = np.array(ikInitGuess)
    
    for idx in range(pos_traj.shape[0]):
        joint_angles = ik_solver.solve(
            target_pos=pos_traj[idx],
            initial_angles=ikInitGuess
        )
        joint_traj[:, idx] = joint_angles
        ikInitGuess = joint_angles  # Warm start for next iteration
    
    return joint_traj

# ====================== Bowl Trajectories ======================
bowl_trajectories = {
    'one': {
        'waypoints': [
            [0.29014 - ee_offset_x, -0.02273, 0.2457],
            [0.35 - ee_offset_x, 0.068, 0.050],
            [0.35 - ee_offset_x, 0.068, 0.030],
            [0.35 - ee_offset_x, 0.068, 0.050],
            [0.29014 - ee_offset_x, -0.02273, 0.2457]
        ],
        'orientations': [[0,0,0]]*5
    },
    'two': {
        'waypoints': [
            [0.29014 - ee_offset_x, -0.02273, 0.2457],
            [0.37 - ee_offset_x, -0.023, 0.050],
            [0.37 - ee_offset_x, -0.023, 0.030],
            [0.37 - ee_offset_x, -0.023, 0.050],
            [0.29014 - ee_offset_x, -0.02273, 0.2457]
        ],
        'orientations': [[0,0,0]]*5
    },
    'three': {
        'waypoints': [
            [0.29014 - ee_offset_x, -0.02273, 0.2457],
            [0.35 - ee_offset_x, -0.114, 0.050],
            [0.35 - ee_offset_x, -0.114, 0.030],
            [0.35 - ee_offset_x, -0.114, 0.050],
            [0.29014 - ee_offset_x, -0.02273, 0.2457]
        ],
        'orientations': [[0,0,0]]*5
    }
}

# Precompute trajectories
print("Precomputing trajectories...")
precomputed_trajectories = {}
for cmd in classes:
    pos_traj, _ = generate_trajectory(
        bowl_trajectories[cmd]['waypoints'],
        bowl_trajectories[cmd]['orientations'],
        10
    )
    precomputed_trajectories[cmd] = calculate_ik_trajectory(pos_traj)

# ====================== Real-time Control Section ======================
command_queue = queue.Queue()
audio_buffer = np.array([])
prediction_threshold = 0.65

class ArmController:
    def init(self):
        self.current_trajectory = None
        self.start_time = 0
        self.running = False
        
    def execute(self, command):
        if not self.running and command in precomputed_trajectories:
            self.current_trajectory = precomputed_trajectories[command]
            self.start_time = time.time()
            self.running = True
            print(f"Starting trajectory for bowl {command}")
            
    def update(self):
        if not self.running:
            return
        
        elapsed = time.time() - self.start_time
        total_time = 10
        idx = min(int((elapsed / total_time) * self.current_trajectory.shape[1]), 
                 self.current_trajectory.shape[1]-1)
        
        p.setJointMotorControlArray(
            robot_id, range(numJoints), p.POSITION_CONTROL,
            targetPositions=self.current_trajectory[:, idx],
            forces=[100]*numJoints,
            positionGains=[0.2]*numJoints
        )
        
        if elapsed >= total_time:
            self.running = False
            print("Trajectory completed")

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = np.concatenate((audio_buffer, indata[:, 0]))
    
    if len(audio_buffer) >= SAMPLE_RATE:
        chunk = audio_buffer[:SAMPLE_RATE]
        audio_buffer = audio_buffer[SAMPLE_RATE:]
        
        if np.max(np.abs(chunk)) < 0.01:
            return
        
        mfcc = librosa.feature.mfcc(y=chunk, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc = librosa.util.fix_length(mfcc, size=64, axis=1)
        mfcc = np.resize(mfcc, (64, 64))
        mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc) + 1e-8)
        
        pred = model.predict(mfcc.reshape(1, 64, 64, 1), verbose=0)[0]
        if np.max(pred) > prediction_threshold:
            command = label_encoder.inverse_transform([np.argmax(pred)])[0]
            command_queue.put(command)

try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=2048
    )
    stream.start()
except Exception as e:
    print(f"Failed to start audio stream: {e}")
    p.disconnect()
    exit()

controller = ArmController()

try:
    print("Voice control active. Say 'one', 'two', or 'three'...")
    while True:
        try:
            cmd = command_queue.get_nowait()
            print(f"Executing command: {cmd}")
            controller.execute(cmd)
        except queue.Empty:
            pass
        
        controller.update()
        p.stepSimulation()
        time.sleep(1/240)

except KeyboardInterrupt:
    print("\nShutting down gracefully...")
    stream.stop()
    stream.close()
    p.disconnect()
except Exception as e:
    print(f"Error occurred: {e}")
    stream.stop()
    stream.close()
    p.disconnect()