import os
import sys
import csv # Added for CSV logging
import time # Added for time.sleep
from grid import Tetromino
GAME_TYPE = 'regular'

pause_time = 0.01 # Define pause_time for AI watching

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from game import Game
import threading
import random
import pickle
from setup import *
from gui import Gui
import multiprocessing

def worker_main(model_filename, target_size, max_steps_per_episode, proc_num, queue):

    model = keras.models.load_model(model_filename)
    data, avg_score = get_data_from_playing_search(model, target_size=target_size, max_steps_per_episode=max_steps_per_episode, proc_num=proc_num)
    queue.put((data, avg_score), block=False)

shape_main_grid = (-1, GAME_BOARD_HEIGHT, GAME_BOARD_WIDTH, 1)
if STATE_INPUT == 'short':
    shape_hold_next = (1, 1 * 2 + 1 + 1 + 6 * Tetromino.pool_size())
    shape_hold_next_description = '[height_sum, hole_sum, combo, is_hold, 6 * 7 type] -> length = 43'
    split_hold_next = 1 * 2 + 1 + 1
else:
    shape_hold_next = (1, GAME_BOARD_WIDTH * 2 + 1 + 6 * Tetromino.pool_size())
    split_hold_next = GAME_BOARD_WIDTH * 2 + 1

shape_dense = (1, GAME_BOARD_WIDTH * 2 + 1 + 6 * Tetromino.pool_size())

gamma = 0.95
epsilon = 0.06

current_avg_score = 0
rand = random.Random()

penalty = -500
# reward_coef = [1.0, 0.5, 0.3, 0.2]
reward_coef = [1.0, 1.0, 1.0, 1.0]
# reward_coef_plan = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], 1, 50]
reward_coef_plan = [[1.0, 1.0, 1.0, 1.0], [1.5, 1.2, 1.0, 0.8], 1, 50] # Updated plan
num_search_best = 6
num_search_rd = 6
env_debug = None


def make_model_conv2d_v1():
    main_grid_input = keras.Input(shape=shape_main_grid[1:], name="main_grid_input")
    a = layers.Conv2D(
        64, 6, activation="relu", input_shape=shape_main_grid[1:]
    )(main_grid_input)
    a = layers.Conv2D(32, (3, 3), activation="relu")(a)
    a = layers.MaxPool2D(pool_size=(13, 3))(a)
    a = layers.Flatten()(a)

    b = layers.Conv2D(
        128, 4, activation="relu", input_shape=shape_main_grid[1:]
    )(main_grid_input)
    b = layers.Conv2D(32, (3, 3), activation="relu")(b)
    b = layers.MaxPool2D(pool_size=(15, 5))(b)
    b = layers.Flatten()(b)

    hold_next_input = keras.Input(shape=shape_hold_next[1:], name="hold_next_input")

    x = layers.concatenate([a, b, hold_next_input])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    critic_output = layers.Dense(1)(x)  # activation=None -> 'linear'

    model_new = keras.Model(
        inputs=[main_grid_input, hold_next_input],
        outputs=critic_output
    )

    model_new.summary()

    return model_new


def make_model_conv2d_v0():
    main_grid_input = keras.Input(shape=shape_main_grid[1:], name="main_grid_input")
    a = layers.Conv2D(
        128, 6, activation="relu", input_shape=shape_main_grid[1:]
    )(main_grid_input)
    a1 = layers.MaxPool2D(pool_size=(15, 5), strides=(1, 1))(a)
    a1 = layers.Flatten()(a1)
    a2 = layers.AvgPool2D(pool_size=(15, 5))(a)
    a2 = layers.Flatten()(a2)

    b = layers.Conv2D(
        256, 4, activation="relu", input_shape=shape_main_grid[1:]
    )(main_grid_input)
    b1 = layers.MaxPool2D(pool_size=(17, 7), strides=(1, 1))(b)
    b1 = layers.Flatten()(b1)
    b2 = layers.AvgPool2D(pool_size=(17, 7))(b)
    b2 = layers.Flatten()(b2)

    hold_next_input = keras.Input(shape=shape_hold_next[1:], name="hold_next_input")

    x = layers.concatenate([a1, a2, b1, b2, hold_next_input])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    critic_output = layers.Dense(1)(x)  # activation=None -> 'linear'

    model_new = keras.Model(
        inputs=[main_grid_input, hold_next_input],
        outputs=critic_output
    )

    model_new.summary()

    return model_new


def make_model_dense():
    dense_input = keras.Input(shape=shape_dense[1:], name="input")

    x = layers.Dense(256, activation="relu")(dense_input)
    x = layers.Dense(128, activation="relu")(x)
    critic_output = layers.Dense(1)(x)  # activation=None -> 'linear'

    model_new = keras.Model(
        inputs=dense_input,
        outputs=critic_output
    )

    model_new.summary()

    return model_new


def load_model(filepath=None):
    if STATE_INPUT == 'short' or STATE_INPUT == 'long':
        model_loaded = make_model_conv2d_v1()
    elif STATE_INPUT == 'dense':
        model_loaded = make_model_dense()
    else:
        model_loaded = None
        sys.stderr.write('STATE_INPUT is wrong. Exit...')
        exit()

    model_loaded.compile(
        optimizer=keras.optimizers.Adam(0.001),
        # loss='huber_loss',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    if filepath is None:
        from pathlib import Path
        Path(FOLDER_NAME + "whole_model").mkdir(parents=True, exist_ok=True)
        Path(FOLDER_NAME + "checkpoints_dqn").mkdir(parents=True, exist_ok=True)
        model_loaded.save(f"{FOLDER_NAME}whole_model/outer_0.keras") 
        model_loaded.save_weights(f"{FOLDER_NAME}checkpoints_dqn/outer_0.weights.h5")  # Save weights
        print('model initial state has been saved')


    return model_loaded


def ai_play(model, max_games=100, mode='piece', is_gui_on=True):
    max_steps_per_episode = 2000
    seed = None
    gui = Gui() if is_gui_on else None
    env = Game(gui=gui, seed=seed, height=0)

    episode_count = 0
    total_score = 0

    pause_time = 0.00

    log_filename = os.path.join(FOLDER_NAME, 'ai_play_log.csv')
    log_file_exists = os.path.isfile(log_filename)

    with open(log_filename, 'a', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        if not log_file_exists:
            log_writer.writerow(['Timestamp', 'Seed', 'GameNumber', 'FinalScore', 'LinesCleared', 'Tetrises', 'TSD', 'TSM', 'TSS', 'MaxCombo', 'PiecesPlaced', 'SessionHighScore'])
        
        for episode_count in range(max_games):
            env.reset()
            current_seed = env.current_state.seed
            high_score = 0 # Initialize high score for this session
            start_time = time.time()

            for step in range(max_steps_per_episode):
                states, add_scores, dones, _, _, moves, _ = env.get_all_possible_states_input()
                rewards = get_reward(add_scores, dones)
                q = rewards + model(split_input(states))
                best = tf.argmax(q).numpy()[0]

                if mode == 'step':
                    best_moves = moves[best]

                    for i in range(len(best_moves) - 1):
                        move = best_moves[i]
                        env.step(action=move)
                        env.render()
                        time.sleep(pause_time)
                    env.step(chosen=best)
                    env.render()
                    time.sleep(pause_time)
                else:
                    env.step(chosen=best)
                    env.render()
                    time.sleep(pause_time)

                if env.is_done() or step == max_steps_per_episode - 1:
                    episode_count += 1
                    final_score = env.current_state.score
                    total_score += final_score
                    lines_cleared = env.current_state.lines_cleared
                    tetrises = env.current_state.tetrises
                    tsd = env.current_state.tsd
                    tsm = env.current_state.tsm
                    tss = env.current_state.tss
                    max_combo = env.current_state.max_combo
                    pieces_placed = env.current_state.pieces_placed

                    if final_score > high_score:
                        high_score = final_score
                    
                    print('episode #{}:   score:{}'.format(episode_count, final_score))
                    log_writer.writerow([
                        start_time, 
                        current_seed, 
                        episode_count, 
                        final_score, 
                        lines_cleared, 
                        tetrises,
                        tsd,
                        tsm,
                        tss,
                        max_combo,
                        pieces_placed,
                        high_score # Added session high score
                    ])
                    csvfile.flush() # Ensure data is written immediately
                    break

    print('average score = {:7.2f}'.format(total_score / max_games))


def ai_play_search(model, max_games=100, is_gui_on=True):
    max_steps_per_episode = 2000
    seed = None
    gui = Gui() if is_gui_on else None
    env = Game(seed=seed, height=0)
    env_gui = Game(gui=gui)

    episode_count = 0
    total_score = 0

    pause_time = 0.04

    log_filename = os.path.join(FOLDER_NAME, 'ai_play_search_log.csv')
    log_file_exists = os.path.isfile(log_filename)

    with open(log_filename, 'a', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        if not log_file_exists:
            log_writer.writerow(['Timestamp', 'Seed', 'GameNumber', 'FinalScore', 'LinesCleared', 'Tetrises', 'TSD', 'TSM', 'TSS', 'MaxCombo', 'PiecesPlaced', 'SessionHighScore'])

        for episode_count in range(max_games):
            env.reset()
            old_state = env.current_state.copy()
            moves_buffer = []
            current_seed = env.current_state.seed
            high_score = 0 # Initialize high score for this session
            start_time = time.time()

            for step in range(max_steps_per_episode):
                env_gui.current_state = old_state.copy()
                old_state = env.current_state.copy()
                moves = []
                thread = threading.Thread(target=ai_get_moves, args=(model, env, moves))
                thread.start()

                for m in moves_buffer:
                    env_gui.step(action=m)
                    env_gui.render()
                    time.sleep(pause_time)

                thread.join()
                moves_buffer = moves

                if env.current_state.game_status == 'gameover':
                    break

                if env.is_done() or step >= max_steps_per_episode - 1:
                    episode_count += 1
                    final_score = env.current_state.score
                    total_score += final_score
                    lines_cleared = env.current_state.lines_cleared
                    tetrises = env.current_state.tetrises
                    tsd = env.current_state.tsd
                    tsm = env.current_state.tsm
                    tss = env.current_state.tss
                    max_combo = env.current_state.max_combo
                    pieces_placed = env.current_state.pieces_placed

                    if final_score > high_score:
                        high_score = final_score

                    print('episode #{}:   score:{}'.format(episode_count, final_score))
                    log_writer.writerow([
                        start_time, 
                        current_seed, 
                        episode_count, 
                        final_score, 
                        lines_cleared, 
                        tetrises,
                        tsd,
                        tsm,
                        tss,
                        max_combo,
                        pieces_placed,
                        high_score # Added session high score
                    ])
                    csvfile.flush()
                    break
    print('average score = {:7.2f}'.format(total_score / max_games))


def ai_get_moves(model, env, moves):
    gamestates_new, gamestates_steps, reward_prev = search_steps(model, env, num_remain=10, num_random=0, action_take=1)
    moves.clear()
    moves += env.get_moves(gamestates_steps[0][0])
    env.current_state = gamestates_steps[0][0]


def search_steps(model, env, num_remain=num_search_best, num_random=num_search_rd, action_take=1):
    gamestates_new, gamestates_steps, reward_prev = search_one_step(model, [env.current_state], env,
                                                                    num_to_choose=num_remain, num_random=num_random)

    save = [[], [], []]
    if action_take == 1:
        save = gamestates_new[-1], gamestates_steps[-1], reward_prev[-1]

    for _ in range(3):
        gamestates_new, gamestates_steps, reward_prev = search_one_step(model, gamestates_new, env,
                                                                        gamestates_steps_old=gamestates_steps,
                                                                        reward_prev_old=reward_prev,
                                                                        num_to_choose=num_remain, num_random=num_random)

    if action_take != 1:
        gamestates_new, gamestates_steps, reward_prev = search_one_step(model, gamestates_new, env,
                                                                        gamestates_steps_old=gamestates_steps,
                                                                        reward_prev_old=reward_prev, num_to_choose=1,
                                                                        num_random=1)
        return gamestates_new, gamestates_steps, reward_prev
    else:
        gamestates_new, gamestates_steps, reward_prev = search_one_step(model, gamestates_new, env,
                                                                        gamestates_steps_old=gamestates_steps,
                                                                        reward_prev_old=reward_prev, num_to_choose=1,
                                                                        num_random=0)
        gamestates_new = [gamestates_steps[0][0], save[0]]
        gamestates_steps = [gamestates_steps[0][:1], save[1]]
        reward_prev = [reward_prev[0], save[2]]
        return gamestates_new, gamestates_steps, reward_prev


def search_one_step(model, gamestates_old, env, gamestates_steps_old=None, reward_prev_old=None, num_to_choose=10,
                    num_random=5):
    s_all = list()
    r_all = list()
    done_all = list()
    gamestates_new = list()
    gamestates_steps_new = list()

    if gamestates_steps_old is None:
        gamestates_steps_old = [[]] * len(gamestates_old)

    if reward_prev_old is None:
        # Ensure reward_prev_old is a list of scalars if initialized here
        reward_prev_old = np.array([0.0] * len(gamestates_old)) # Use float for consistency

    for i in range(len(gamestates_old)):
        states, add_scores, dones, _, _, _, gamestates = env.get_all_possible_states_input(gamestates_old[i])
        s_all.append(states)
        
        current_reward_prev_item = reward_prev_old[i]
        actual_prev_reward = 0.0 # Default
        if isinstance(current_reward_prev_item, (int, float)):
            actual_prev_reward = float(current_reward_prev_item)
        elif isinstance(current_reward_prev_item, (np.ndarray, list, tuple)):
            if len(current_reward_prev_item) == 1:
                try:
                    actual_prev_reward = float(current_reward_prev_item[0])
                except TypeError: # Handle cases like np.array(None) if that could occur
                    print(f"Warning: Could not convert element of reward_prev_old[i] to float: {current_reward_prev_item[0]}. Using 0.")
                    actual_prev_reward = 0.0
            elif not current_reward_prev_item: # Empty
                actual_prev_reward = 0.0
            else: # Multiple elements, should ideally not happen for a single reward value
                print(f"Warning: Unexpected multi-element reward_prev_old[i]: {current_reward_prev_item}. Using first element if possible, else 0.")
                try:
                    actual_prev_reward = float(current_reward_prev_item[0])
                except (IndexError, TypeError):
                    actual_prev_reward = 0.0
        else: # Other unexpected types
            print(f"Warning: Unexpected type for reward_prev_old[i]: {type(current_reward_prev_item)}, value: {current_reward_prev_item}. Using 0.")
            actual_prev_reward = 0.0
            
        r_all.append(get_reward(add_scores, dones, add=actual_prev_reward / gamma))
        done_all += dones
        gamestates_new += gamestates
        for j in range(len(gamestates)):
            gamestates_steps_new.append(gamestates_steps_old[i].copy() + [gamestates[j]])

    s_all = np.concatenate(s_all)
    r_all = np.concatenate(r_all)
    q = model(split_input(s_all)) + r_all

    arg_sorted = tf.argsort(tf.reshape(q, -1), direction='DESCENDING').numpy().tolist()
    gamestates_chosen = []
    reward_prev_chosen = []
    gamestates_steps_chosen = []

    prev = 0
    q_prev = None
    num_to_choose = min(num_to_choose, len(gamestates_new))

    for _ in range(num_to_choose):
        while prev < len(arg_sorted) and q[arg_sorted[prev]] == q_prev:
            prev += 1
        if prev >= len(arg_sorted):
            break

        idx = arg_sorted[prev]
        gamestates_chosen.append(gamestates_new[idx])
        reward_prev_chosen.append(r_all[idx])
        gamestates_steps_chosen.append(gamestates_steps_new[idx])
        q_prev = q[idx]
        prev += 1

    num_random = min(num_random, len(gamestates_new))
    for _ in range(num_random):
        rd_int = random.randint(0, len(gamestates_new) - 1)
        gamestates_chosen.append(gamestates_new[rd_int])
        reward_prev_chosen.append(r_all[rd_int])
        gamestates_steps_chosen.append(gamestates_steps_new[rd_int])

    return gamestates_chosen, gamestates_steps_chosen, reward_prev_chosen


def split_input(states):
    if STATE_INPUT == 'dense':
        return states
    else:
        in1, in2 = tf.split(states, [GAME_BOARD_HEIGHT * GAME_BOARD_WIDTH, -1], axis=1)
        return tf.reshape(in1, shape_main_grid), in2


def gamestates_to_training_data(env, gamestates_steps):
    row_data = list()

    gamestate_prev = env.current_state
    for i in range(len(gamestates_steps)):
        s_ = env.get_state_input(gamestate_prev)
        sp_ = env.get_state_input(gamestates_steps[i])
        add_score_ = gamestates_steps[i].score - gamestate_prev.score
        done = gamestates_steps[i].game_status == 'gameover'
        row_data.append((s_, sp_, add_score_, done))
        gamestate_prev = gamestates_steps[i]
        if done:
            break

    return row_data


def get_data_from_playing_cnn2d(model_filename, target_size=8000, max_steps_per_episode=2000, proc_num=0,
                                queue=None):
    tf.autograph.set_verbosity(3)
    model = keras.models.load_model(model_filename)
    if model is None:
        print('ERROR: model has not been loaded. Check this part.')
        exit()

    global epsilon
    if proc_num == 0:
        epsilon = 0

    data = list()
    env = Game()
    episode_max = 1000
    total_score = 0
    avg_score = 0
    t_spins = 0

    for episode in range(episode_max):
        # env.reset(rand.randint(0, 10))
        env.reset()
        episode_data = list()
        for step in range(max_steps_per_episode):
            s = env.get_state_input(env.current_state)
            possible_states, add_scores, dones, is_include_hold, is_new_hold, _, _ = env.get_all_possible_states_input()
            rewards = get_reward(add_scores, dones)

            pool_size = Tetromino.pool_size()

            # get the best first before modifying the last next
            q = rewards + model(split_input(possible_states), training=False).numpy()
            for j in range(len(dones)):
                if dones[j]:
                    q[j] = rewards[j]
            best = tf.argmax(q).numpy()[0] + 0

            # if hold was empty, then we don't know what's next; if hold was not empty, then we know what's next!
            if is_include_hold and not is_new_hold:
                possible_states[1][:-1, -pool_size:] = 0
            else:
                possible_states[1][:, -pool_size:] = 0

            rand_fl = rand.random()
            if rand_fl > epsilon:
                chosen = best
            else:
                # probability based on q
                # q_normal = q.reshape(-1)
                # q_normal = q_normal - np.min(q_normal) + 0.001
                # q_normal = q_normal / np.sum(q_normal) + 0.3
                # q_normal = q_normal / np.sum(q_normal)
                # chosen = np.random.choice(q_normal.shape[0], p=q_normal)

                # uniform probability
                chosen = random.randint(0, len(dones) - 1)

            episode_data.append(
                (s, (possible_states[0][best], possible_states[1][best]), add_scores[best], dones[best]))

            if add_scores[best] != int(add_scores[best]):
                t_spins += 1

            env.step(chosen=chosen)

            if env.is_done() or step == max_steps_per_episode - 1:
                data += episode_data
                total_score += env.current_state.score
                break

        if len(data) > target_size:
            print('proc_num: #{:<2d} | total episodes:{:<4d} | avg score:{:<7.2f} | data size:{} | t-spins: {}'.format(
                proc_num, episode + 1, total_score / (episode + 1), len(data), t_spins))
            avg_score = total_score / (episode + 1)
            break

    if queue is not None:
        queue.put((data, avg_score), block=False)
        return

    return data, avg_score


def get_data_from_playing_search(model, target_size=8000, max_steps_per_episode=1000, proc_num=0,
                                 queue=None):
    tf.autograph.set_verbosity(3)

    global epsilon
    if proc_num == 0:
        epsilon = 0

    data = list()
    env = Game()
    episode_max = 1000
    total_score = 0
    avg_score = 0

    for episode in range(episode_max):
        env.reset()
        episode_data = list()
        for step in range(int(max_steps_per_episode)):
            gamestates_new, gamestates_steps, reward_prev = search_steps(model, env, action_take=5)
            episode_data += gamestates_to_training_data(env, gamestates_steps[0])

            if rand.random() > epsilon:
                env.current_state = gamestates_new[0].copy()
            else:
                env.current_state = gamestates_new[-1].copy()

            if env.is_done() or len(data) + len(episode_data) >= target_size:
                break

            if proc_num == 0:
                sys.stdout.write(
                    f'\r data: {len(data) + len(episode_data)} / {target_size} |'
                    f' score per step : {(total_score + env.current_state.score) / (len(data) + len(episode_data)):<6.2f} |'
                    f' game num : {episode + 1}')
                sys.stdout.flush()

        data += episode_data
        total_score += env.current_state.score

        if len(data) >= target_size:
            if proc_num == 0:
                print('\n proc_num: #{:<2d} | total episodes:{:<4d} | avg score:{:<7.2f} | data size:{}'.format(
                    proc_num, episode + 1, total_score / (episode + 1), len(data)))
            avg_score = total_score / (episode + 1)
            break

    if queue is not None:
        queue.put((data, avg_score), block=False)
        return

    return data, avg_score


def train(model, outer_start=0, outer_max=100):
    inner_max = 5
    epoch_training = 5
    batch_training = 512

    buffer_new_size = 12000
    buffer_outer_max = 4
    repeat_new_buffer = 2
    history = None
    global gamma # Ensure gamma is accessible

    for outer in range(outer_start + 1, outer_start + 1 + outer_max):
        print('-- outer loop # {} --'.format(outer))
        time_outer_begin = time.time()
        modify_reward_coef(outer)

        buffer = []

        new_buffer = collect_samples_multiprocess_queue(model_filename=FOLDER_NAME + f'whole_model/outer_{outer - 1}.keras',
                                                        target_size=buffer_new_size)
        save_buffer_to_file(FOLDER_NAME + f'dataset/buffer_{outer}.pkl', new_buffer)
        buffer += new_buffer

        for i in range(max(1, outer - buffer_outer_max + 1), outer):
            buffer += load_buffer_from_file(filename=FOLDER_NAME + 'dataset/buffer_{}.pkl'.format(i))

        for _ in range(repeat_new_buffer):
            buffer += load_buffer_from_file(filename=FOLDER_NAME + 'dataset/buffer_{}.pkl'.format(outer))

        random.shuffle(buffer)

        s, s_, r_, dones_ = process_buffer_best(buffer)

        buffer_size = len(buffer)
        new_buffer_size = len(new_buffer)
        del buffer
        del new_buffer

        for inner in range(inner_max):
            print(f"      -- inner # {inner + 1}/{inner_max} --")
            
            # Corrected target calculation logic
            q_s_prime_values = []
            # Calculate Q(s') in batches if s_ is large, or directly if manageable
            # Assuming s_ fits in memory for model prediction for simplicity here.
            # If s_ is very large, batching this prediction is necessary.
            # For now, let's assume direct prediction is fine as per original structure attempt.
            
            # Batched prediction for Q(s')
            q_s_prime_batches = []
            for i_batch in range(int(s_.shape[0] / batch_training) + 1):
                start_batch = i_batch * batch_training
                end_batch = min((i_batch + 1) * batch_training, s_.shape[0])
                if start_batch >= end_batch:
                    continue
                q_s_prime_batch = model(split_input(s_[start_batch:end_batch]), training=False).numpy().reshape(-1)
                q_s_prime_batches.append(q_s_prime_batch)
            
            if not q_s_prime_batches: # Handle empty s_ case if it occurs
                if s_.shape[0] == 0:
                    print("Warning: s_ is empty, skipping inner loop.")
                    continue 
                else: # Should not happen if loop above runs for non-empty s_
                    raise ValueError("q_s_prime_batches is empty but s_ was not.")

            q_s_prime_full = np.concatenate(q_s_prime_batches)
            
            calculated_target = np.zeros_like(q_s_prime_full)

            for i in range(len(dones_)): # dones_ corresponds to s, s_ pairs
                if dones_[i]:
                    calculated_target[i] = r_[i]  # Target is immediate reward if state is terminal
                else:
                    calculated_target[i] = r_[i] + gamma * q_s_prime_full[i]  # Target is r + gamma * Q(s')

            if inner == inner_max - 1:
                save_training_dataset_to_file(filename=FOLDER_NAME + 'dataset/dataset_{}.pkl'.format(outer),
                                              dataset=(s, calculated_target)) # Save s and correct target

            history = model.fit(split_input(s), calculated_target, batch_size=batch_training, epochs=epoch_training, verbose=0)
            print('      loss = {:8.3f}   mse = {:8.3f}'.format(history.history['loss'][-1],
                                                                history.history['mean_squared_error'][-1]))

        model.save(f"{FOLDER_NAME}whole_model/outer_{outer}.keras")
        model.save_weights(f"{FOLDER_NAME}checkpoints_dqn/outer_{outer}.weights.h5")

        time_outer_end = time.time()
        text_ = ''
        if outer == 1:
            text_ += f'input shapes: {shape_main_grid} {shape_hold_next} \n {shape_hold_next_description} \n'

        text_ += 'outer = {:>4d} | pre-training avg score = {:>8.3f} | loss = {:>8.3f} | mse = {:>8.3f} |' \
                 ' dataset size = {:>7d} | new dataset size = {:>7d} | time elapsed: {:>6.1f} sec | coef = {} | penalty = {:>7d} | gamma = {:>6.3f} |' \
                 ' search best/rd = {}, {} |\n' \
            .format(outer, current_avg_score, history.history['loss'][-1], history.history['mean_squared_error'][-1],
                    buffer_size, new_buffer_size, time_outer_end - time_outer_begin, reward_coef, penalty, gamma,
                    num_search_best, num_search_rd
                    )
        append_record(text_)
        print('   ' + text_)


def save_buffer_to_file(filename, buffer):
    from pathlib import Path
    Path(FOLDER_NAME + 'dataset').mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(buffer, f)


def save_training_dataset_to_file(filename, dataset):
    from pathlib import Path
    Path(FOLDER_NAME + 'dataset').mkdir(parents=True, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def load_buffer_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def process_buffer_best(buffer):
    s = list()
    s_ = list()
    add_scores = list()
    dones_ = list()
    for row in buffer:
        s.append(row[0])
        s_.append(row[1])
        add_scores.append(row[2])
        dones_ += [row[3]]

    s = np.concatenate(s)
    s_ = np.concatenate(s_)
    r_ = get_reward(add_scores, dones_)
    r_ = np.concatenate(r_)
    return s, s_, r_, dones_


def render_env_debug_state_input(state):
    if STATE_INPUT == 'dense':
        return

    global env_debug
    if env_debug is None:
        env_debug = Game(gui=Gui())

    loc = 0
    for r in range(GAME_BOARD_HEIGHT):
        for c in range(GAME_BOARD_WIDTH):
            env_debug.current_state.grid[r][c] = state[0, loc]
            loc += 1

    if STATE_INPUT == 'short':
        loc += 3
    else:
        loc += 21

    env_debug.render()


def render_env_debug_gamestate(gamestate):
    if STATE_INPUT == 'dense':
        return

    global env_debug
    if env_debug is None:
        env_debug = Game(gui=Gui())

    env_debug.current_state = gamestate
    env_debug.render()


def get_q_from_gamestate(model, gamestate):
    return model(split_input(Game.get_state_input(gamestate))).numpy()


def check_same_state(s1, s2):
    s1_ = s1.reshape(-1)
    s2_ = s2.reshape(-1)
    for i in range(s1_.shape[0]):
        if s1_[i] != s2_[i]: return False

    return True


def append_record(text, filename=None):
    if filename is None:
        filename = FOLDER_NAME + 'record.txt'
    with open(filename, 'a') as f:
        f.write(text)


def collect_samples_multiprocess_queue(model_filename, target_size=10000):
    multiprocessing.freeze_support()    # Windows-safe
    cpu_count = min(multiprocessing.cpu_count(), CPU_MAX)
    q = multiprocessing.Queue()
    jobs = []

    for i in range(cpu_count):
        p = multiprocessing.Process(
            target=worker_main,
            args=(model_filename,
                  int(target_size/cpu_count),
                  1000,           # or whatever your max_steps is
                  i, q)
        )
        p.start()
        jobs.append(p)

    # gather results
    data, scores = [], []
    for _ in jobs:
        d, s = q.get(timeout=7200)
        data += d
        scores.append(s)

    for p in jobs:
        p.join()

    print(f'end multiprocess: total data length: {len(data)} | avg score: {max(scores):<7.2f}')
    return data


def modify_reward_coef(outer):
    global reward_coef
    r_1 = reward_coef_plan[0]
    r_2 = reward_coef_plan[1]
    start = reward_coef_plan[2]
    end = reward_coef_plan[3]
    for i in range(len(reward_coef)):
        rate = (outer - start) / (end - start)
        rate = min(rate, 1)
        rate = max(rate, 0)
        reward_coef[i] = r_1[i] + (r_2[i] - r_1[i]) * rate
        reward_coef[i] = round(reward_coef[i] * 1024) / 1024
    print(f' reward_coef modified to {reward_coef}')


def get_reward(add_scores, dones, add=0): # add is now always scalar
    reward = []
    # Assuming add_scores and dones are iterable and of the same length.
    for i in range(len(add_scores)):
        current_add_score_item = add_scores[i]
        current_done = dones[i]
        
        actual_add_score = 0.0 # Default
        if isinstance(current_add_score_item, (int, float)):
            actual_add_score = float(current_add_score_item)
        elif isinstance(current_add_score_item, (list, tuple, np.ndarray)):
            if len(current_add_score_item) == 1:
                try:
                    actual_add_score = float(current_add_score_item[0])
                except TypeError:
                    print(f"Warning: Could not convert element of add_scores to float: {current_add_score_item[0]}. Using 0.")
                    actual_add_score = 0.0
            elif not current_add_score_item: # Empty list/array
                actual_add_score = 0.0
            else: # Multiple elements
                print(f"Warning: Unexpected multi-element score in add_scores: {current_add_score_item}. Using sum: {sum(current_add_score_item)}.")
                try:
                    actual_add_score = float(sum(current_add_score_item))
                except TypeError:
                    print(f"Warning: Could not sum elements of add_scores: {current_add_score_item}. Using 0.")
                    actual_add_score = 0.0
        else: # Other unexpected types
            print(f"Warning: Unexpected type for score item: {type(current_add_score_item)}, value: {current_add_score_item}. Using 0.")
            actual_add_score = 0.0

        current_reward_value = 0.0 # This will be scalar

        if current_done:
            current_reward_value = float(penalty)  # Apply penalty for game over
        else:
            # Apply reward_coef based on score thresholds using actual_add_score (scalar)
            if actual_add_score >= 90:
                current_reward_value = actual_add_score * reward_coef[0]
            elif actual_add_score >= 50:
                current_reward_value = actual_add_score * reward_coef[1]
            elif actual_add_score >= 20:
                current_reward_value = actual_add_score * reward_coef[2]
            elif actual_add_score > 0:
                current_reward_value = actual_add_score * reward_coef[3]
            # If actual_add_score is 0 (no lines cleared), current_reward_value remains 0.0.
            
        # Ensure current_reward_value is float before adding
        current_reward_value = float(current_reward_value)
            
        reward.append(current_reward_value + add) # scalar + scalar = scalar
    
    # reward is now a list of scalars.
    if not reward: # Handles case where add_scores was empty
        return np.array([]).reshape([-1, 1])
        
    return np.array(reward, dtype=float).reshape([-1, 1]) # Specify dtype for robustness

def main():

    if MODE == 'human_player':
        game = Game(gui=Gui(), seed=None)
        game.restart()
        game.run()
    elif MODE == 'ai_player_training':
        if OUT_START == 0:
            load_model()
        model_load = keras.models.load_model(FOLDER_NAME + 'whole_model/outer_{}.keras'.format(OUT_START))
        train(model_load, outer_start=OUT_START, outer_max=OUTER_MAX)
    elif MODE == 'ai_player_watching':
        model_load = keras.models.load_model(FOLDER_NAME + 'whole_model/outer_{}.keras'.format(OUT_START))
        ai_play_search(model_load, is_gui_on=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()