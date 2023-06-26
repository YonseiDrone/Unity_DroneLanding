from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import cv2
import tensorflow as tf
import numpy as np
import yaml
from collections import deque
from model import SoftActorCritic
from utils import ReplayBuffer

RANDOM_ACTION = False
USE_ARUCO = False
MODEL_PATH = "model"
MODEL_NAME = "DroneLanding"
ACTION_SIZE = 4

# load unity environment
engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name='UnityEnv/Drone_navigation.exe')
# env = UnityEnvironment()
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]
print(f'name of behavior: {behavior_name}')
spec = env.behavior_specs[behavior_name]
print(spec)

if USE_ARUCO:
    #Generate Camera Matrix
    w, h = 1920, 1080
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # 사용할 Aruco 딕셔너리 선택
    parameters = cv2.aruco.DetectorParameters_create()
    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

#define
replay = ReplayBuffer(9 if USE_ARUCO else 6, ACTION_SIZE)
writer = tf.summary.create_file_writer(MODEL_PATH + '/' + MODEL_NAME + '/summary')
model = SoftActorCritic(ACTION_SIZE, writer)
total_steps = 0
total_rewards = []

#training
for ep in range(10000):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    tracked_agent = -1
    done = False
    ep_reward = 0
    ep_steps = 0
    last_state = []
    turn_off = False
    confidence = np.zeros((200, 100, 3), np.uint8)
    confidence_deque = deque(maxlen=80)

    while not done:
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]

        # state 받아오기
        if USE_ARUCO:
            cam, vector = decision_steps[tracked_agent].obs
        else:
            vector = decision_steps[tracked_agent].obs[0]

        # action 생성
        if RANDOM_ACTION:
            action = np.random.random(5) * 4 - 2
            action[4] = 0
            action = np.expand_dims(action, axis=0)
            agent_action = ActionTuple()
            agent_action.add_continuous(action)
            env.set_actions(behavior_name, agent_action)
        else:
            if USE_ARUCO:
                cam_gray = np.uint8(cam * 255)
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(cam_gray, aruco_dict, parameters=parameters)
                if ids is not None:
                    id = ids[0][0]
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.42, mtx, dist)
                    print(tvecs[0], vector[3:6])
                    cv2.aruco.drawDetectedMarkers(cam_gray, corners, ids)

                cv2.imshow("Image", cam_gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            action = model.sample_action(np.expand_dims(vector, axis=0))
            action = np.multiply(action, [0.5, 0.5, 0.5, 11])
            if action[3] > 10:
                turn_off = True
                confidence_deque.append(action[3])
            if turn_off:
                print("off")
            else:
                print(action)
            action = np.expand_dims(action, axis=0)
            agent_action = ActionTuple()
            agent_action.add_continuous(action)
            env.set_actions(behavior_name, agent_action)
        env.step()

        # debug confidence
        if not turn_off:
            confidence_deque.append(action[0][3])
        confidence = np.zeros((220, 400, 3), np.uint8)
        confidence_list = list(confidence_deque)
        for i, k in enumerate(confidence_list):
            if i == 0:
                continue
            else:
                cv2.line(confidence, ((i-1)*5, 220-(int((confidence_list[i-1]+11)*10))), ((i*5), 220-(int((confidence_list[i]+11)*10))), (0, 255, 0), 1)
        cv2.line(confidence, (0, 220-210), (400, 220-210), (255, 0, 0), 2)
        cv2.imshow("confidence", confidence)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # next state 받아오기
        reward, next_cam, next_vector = None, None, None
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if tracked_agent in decision_steps:
            reward = decision_steps[tracked_agent].reward
            if USE_ARUCO:
                next_cam, next_vector = decision_steps[tracked_agent].obs
            else:
                next_vector = decision_steps[tracked_agent].obs[0]
        if tracked_agent in terminal_steps:
            reward = terminal_steps[tracked_agent].reward
            if USE_ARUCO:
                next_cam, next_vector = terminal_steps[tracked_agent].obs
            else:
                next_vector = terminal_steps[tracked_agent].obs[0]
            done = True

        ep_reward += reward

        # 버퍼에 저장
        if not done:
            if not turn_off:
                replay.store(vector, action[0], reward, next_vector, done)
            else:
                last_state = [vector, action[0]]
        else:
            if turn_off:
                replay.store(last_state[0], last_state[1], reward, next_vector, done)
            else:
                replay.store(vector, action[0], reward, next_vector, done)

    ep_steps += 1
    total_steps += 1

    if (ep_steps % 1 == 0) and (total_steps > 100):
        for e in range(10):
            vectors, actions, rewards, next_vectors, dones = replay.fetch_sample(16)
            c1_loss, c2_loss, a_loss, alpha_loss = model.train(vectors, actions, rewards, next_vectors, dones)
            print("learn")
            with writer.as_default():
                with writer.as_default():
                    tf.summary.scalar("actor_loss", a_loss, model.epoch_step)
                    tf.summary.scalar("critic1_loss", c1_loss, model.epoch_step)
                    tf.summary.scalar("critic2_loss", c2_loss, model.epoch_step)
                    tf.summary.scalar("alpha_loss", alpha_loss, model.epoch_step)
            model.epoch_step += 1
            if model.epoch_step % 1 == 0:
                model.update_weights()
    if ep % 1 == 0:
        model.policy.save_weights(MODEL_PATH + '/' + MODEL_NAME + '/model')
    total_rewards.append(ep_reward)
    print(f"Episode {ep} reward: {ep_reward}")
    with writer.as_default():
        tf.summary.scalar("episode_reward", ep_reward, ep)

env.close()
