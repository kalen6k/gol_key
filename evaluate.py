# evaluate.py
import os
import argparse
import torch
import gymnasium as gym
from pathlib import Path
import numpy as np
import time
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent_model import GOLKeyAgent
from word_env import WordTargetWrapper

class VLMExtractor(BaseFeaturesExtractor):
    """ Custom feature extractor using the GOLKeyAgent's embed method. """
    def __init__(self, observation_space, agent: "GOLKeyAgent", vlm_internal_batch_size: int):
        super().__init__(observation_space, features_dim=agent.intermediate_state)
        self.agent = agent
        self.proj = agent.proj
        self.vlm_internal_batch_size = vlm_internal_batch_size
        # print(f"VLMExtractor Initialized: Features Dim = {agent.intermediate_state}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raw_features = self.agent.embed(obs, max_batch=self.vlm_internal_batch_size)
        features = self.proj(raw_features.to(torch.float32))
        return features

def load_eval_config():
    parser = argparse.ArgumentParser(description="Evaluate GOLKey Agent (Batched)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_word_file", type=str, default="test_words.txt")
    parser.add_argument("--model_dir_vlm", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--grid_shape", type=int, nargs=2, default=[28, 28])
    parser.add_argument("--env_max_steps", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_csv", type=str, default="evaluation_results_batched.csv")
    parser.add_argument("--vlm_internal_batch_size_eval", type=int, default=4)
    parser.add_argument("--n_eval_envs", type=int, default=8)
    args = parser.parse_args()
    args.GRID_SHAPE = tuple(args.grid_shape)
    return args

def make_env_fn_eval(rank, seed, word_file_for_init, grid_shape, max_steps, vlm_agent_instance):
    def _init():
        env = WordTargetWrapper(
            wordfile=word_file_for_init, min_len=3, max_len=8,
            shape=grid_shape, max_steps=max_steps, agent_instance=vlm_agent_instance
        )
        return env
    return _init

def evaluate_model_vec_batched(config):
    print(f"--- Starting Batched Vectorized Evaluation ---")
    print(f"Model: {config.model_path}, Test Words: {config.test_word_file}, Envs: {config.n_eval_envs}")

    shared_vlm_agent = GOLKeyAgent(model_dir=config.model_dir_vlm)

    with open(config.test_word_file, 'r') as f:
        all_test_words = [line.strip() for line in f if line.strip() and len(line.strip()) >=3 and len(line.strip()) <=8]
    num_total_words_to_test = len(all_test_words)
    if num_total_words_to_test == 0:
        print("No test words. Exiting."); return
    print(f"Loaded {num_total_words_to_test} test words.")
    
    env_fns = [make_env_fn_eval(i, i + 420, config.test_word_file, config.GRID_SHAPE, config.env_max_steps, shared_vlm_agent) for i in range(config.n_eval_envs)]
    eval_vec_env = DummyVecEnv(env_fns)
    eval_vec_env = VecTransposeImage(eval_vec_env)

    if isinstance(eval_vec_env.unwrapped, DummyVecEnv):
        for i in range(config.n_eval_envs):
            eval_vec_env.unwrapped.envs[i].set_eval_mode(True)
    else:
        eval_vec_env.env_method("set_eval_mode", True, indices=list(range(config.n_eval_envs)))

    # policy_kwargs_for_load = dict(
    #     features_extractor_class=VLMExtractor,
    #     features_extractor_kwargs=dict(
    #         agent=shared_vlm_agent,
    #         vlm_internal_batch_size=config.vlm_internal_batch_size_eval
    #     ),
    #     net_arch=dict(pi=[1024, 256, 128, 32],
    #                   vf=[1024, 256, 128, 32]),
    #     ortho_init=(config.device.lower() != "cpu")
    # )
    print("Attempting to load model WITHOUT explicit policy_kwargs...")
    loaded_model = PPO.load(
        config.model_path, 
        device=config.device, 
        env=eval_vec_env, # env is often important for space inference
        custom_objects={
            "learning_rate": 0.0003, "clip_range": 0.2,
        }
    )
    print("PPO model loaded.")
    if hasattr(loaded_model.policy.features_extractor, 'agent') and \
       isinstance(loaded_model.policy.features_extractor, VLMExtractor):
        print("Manually setting agent for loaded VLMExtractor...")
        loaded_model.policy.features_extractor.agent = shared_vlm_agent
        loaded_model.policy.features_extractor.proj = shared_vlm_agent.proj
        loaded_model.policy.features_extractor.vlm_internal_batch_size = config.vlm_internal_batch_size_eval
        expected_features_dim = shared_vlm_agent.intermediate_state
        if loaded_model.policy.features_extractor.features_dim != expected_features_dim:
            print(f"Warning: Loaded VLMExtractor features_dim {loaded_model.policy.features_extractor.features_dim} "
                  f"does not match current agent's intermediate_state {expected_features_dim}. Re-initializing proj might be needed if not done by SB3.")
            
    else:
        print("Warning: Loaded model does not have expected VLMExtractor.")


    if hasattr(loaded_model.policy.features_extractor, 'agent') and \
       isinstance(loaded_model.policy.features_extractor.agent, GOLKeyAgent): # Ensure it's our agent type
        
        vlm_device = loaded_model.policy.features_extractor.agent.device
        ppo_policy_device = loaded_model.policy.device # Device of the PPO policy's networks

        print(f"  INFO: VLM (GOLKeyAgent) is operating on device: {vlm_device}")
        print(f"  INFO: PPO policy networks are on device: {ppo_policy_device}")

        if vlm_device != ppo_policy_device:
            print(f"  WARNING: Device mismatch! VLM device ({vlm_device}) and PPO policy device ({ppo_policy_device}) differ.")
            print(f"           Data will be transferred between devices during forward passes. This is functional but may impact performance.")
            print(f"           - Observations (on {ppo_policy_device}) will be moved to {vlm_device} for VLM embedding.")
            print(f"           - Extracted features (on {vlm_device}) will be moved to {ppo_policy_device} for PPO MLPs.")
        else:
            print(f"  INFO: VLM and PPO policy devices match ({vlm_device}). Ready")
    
    results = []
    next_word_idx = 0
    current_word_for_env = [None] * config.n_eval_envs
    
    current_episode_rewards = np.zeros(config.n_eval_envs, dtype=float)
    current_episode_steps = np.zeros(config.n_eval_envs, dtype=int)


    for i in range(config.n_eval_envs):
        if next_word_idx < num_total_words_to_test:
            target_word = all_test_words[next_word_idx]
            current_word_for_env[i] = target_word
            if isinstance(eval_vec_env.unwrapped, DummyVecEnv):
                eval_vec_env.unwrapped.envs[i].set_next_eval_target(target_word)
            else:
                eval_vec_env.env_method("set_next_eval_target", target_word, indices=[i])
            next_word_idx += 1
        else:
            current_word_for_env[i] = None

    obs = eval_vec_env.reset()
    active_envs_mask = [word is not None for word in current_word_for_env]
    num_active_envs = sum(active_envs_mask)

    print(f"\nInitialization complete. Starting evaluation loop for {num_total_words_to_test} words...")
    print(f"Initial words assigned to envs: {current_word_for_env[:config.n_eval_envs]}")
    print(f"Number of initially active envs: {num_active_envs}\n")

    pbar = tqdm(total=num_total_words_to_test, desc="Evaluating words")

    steps_taken_in_loop = 0
    max_initial_steps_to_log = 5 

    while len(results) < num_total_words_to_test:
        if num_active_envs == 0: break

        if steps_taken_in_loop < max_initial_steps_to_log:
            print(f"  Loop iteration {steps_taken_in_loop + 1}, num_active_envs: {num_active_envs}, results collected: {len(results)}")

        actions, _ = loaded_model.predict(obs, deterministic=True)
        next_obs, step_rewards, dones, infos = eval_vec_env.step(actions)

        for i in range(config.n_eval_envs):
            if not active_envs_mask[i]: continue

            current_episode_rewards[i] += step_rewards[i]
            current_episode_steps[i] += 1


            if dones[i]:
                if steps_taken_in_loop < max_initial_steps_to_log * config.n_eval_envs or len(results) % 10 == 0 : # Log first few dones or every 10th
                    print(f"    Env {i} finished episode for word '{current_word_for_env[i]}'. Success: {infos[i].get('success', False)}. Results: {len(results)+1}/{num_total_words_to_test}")
                processed_word = current_word_for_env[i]
                if processed_word:
                    results.append({
                        'target_word': processed_word,
                        'word_len': len(processed_word),
                        'success': infos[i].get('success', False),
                        'steps': current_episode_steps[i],
                        'reward': current_episode_rewards[i],
                        'final_prefix': infos[i].get('prefix', 0)
                    })
                    pbar.update(1)
                
                current_episode_rewards[i] = 0
                current_episode_steps[i] = 0

                if next_word_idx < num_total_words_to_test:
                    new_target_word = all_test_words[next_word_idx]
                    current_word_for_env[i] = new_target_word
                    if isinstance(eval_vec_env.unwrapped, DummyVecEnv):
                        eval_vec_env.unwrapped.envs[i].set_next_eval_target(new_target_word)
                    else:
                        eval_vec_env.env_method("set_next_eval_target", new_target_word, indices=[i])
                    active_envs_mask[i] = True
                    next_word_idx += 1
                else:
                    current_word_for_env[i] = None
                    active_envs_mask[i] = False
        
        obs = next_obs
        num_active_envs = sum(active_envs_mask)
        if num_active_envs == 0 and len(results) < num_total_words_to_test:
            print("Warning: All active envs finished, but not all words tested.")
            break
            
    pbar.close()
    eval_vec_env.close()

    if not results:
        print("No evaluation results collected."); return
    results_df = pd.DataFrame(results)
    
    print("\n--- Batched Evaluation Summary ---")
    print(results_df.head())
    overall_success_rate = results_df['success'].mean() if not results_df.empty else 0
    mean_episode_reward = results_df['reward'].mean() if not results_df.empty else 0
    mean_episode_steps = results_df['steps'].mean() if not results_df.empty else 0

    print(f"\nOverall Success Rate: {overall_success_rate:.2%}")
    print(f"Mean Episode Reward: {mean_episode_reward:.2f}")
    print(f"Mean Episode Steps: {mean_episode_steps:.2f}")

    if 'word_len' in results_df.columns and not results_df.empty:
        success_by_len = results_df.groupby('word_len')['success'].mean()
        reward_by_len = results_df.groupby('word_len')['reward'].mean()
        steps_by_len = results_df.groupby('word_len')['steps'].mean()
        print("\nSuccess Rate by Word Length:")
        print(success_by_len)
        print("\nMean Reward by Word Length:")
        print(reward_by_len)
        print("\nMean Steps by Word Length:")
        print(steps_by_len)

    results_df.to_csv(config.results_csv, index=False)
    print(f"\nEvaluation results saved to {config.results_csv}")
    print("--- Batched Evaluation Complete ---")

if __name__ == "__main__":
    eval_args = load_eval_config()
    evaluate_model_vec_batched(eval_args)