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
        self.agent_for_embed = agent
        if not hasattr(agent, 'patch_dim') or not hasattr(agent, 'intermediate_state'):
            raise ValueError("The GOLKeyAgent instance provided to VLMExtractor during load "
                             "is missing 'patch_dim' or 'intermediate_state'. "
                             "Ensure GOLKeyAgent unpickles with these attributes.")
            
        self.proj = torch.nn.Linear(agent.patch_dim, agent.intermediate_state)
        self.proj = agent.proj
        self.vlm_internal_batch_size = vlm_internal_batch_size
        print(f"VLMExtractor Initialized. Agent for embed (initially unpickled): {id(self.agent_for_embed)}, "
              f"Proj (created, for SB3 to load weights into): {id(self.proj)}, "
              f"Features Dim: {self.features_dim}")
        
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

    print("Initializing GOLKeyAgent (this will be our 'live' VLM source)...")
    live_vlm_provider = GOLKeyAgent(model_dir=config.model_dir_vlm)
    print(f"Live GOLKeyAgent initialized. ID: {id(live_vlm_provider)}, Device: {live_vlm_provider.device}")

    with open(config.test_word_file, 'r') as f:
        all_test_words = [line.strip() for line in f if line.strip() and len(line.strip()) >=3 and len(line.strip()) <=8]
    num_total_words_to_test = len(all_test_words)
    if num_total_words_to_test == 0:
        print("No test words. Exiting."); return
    print(f"Loaded {num_total_words_to_test} test words.")
    
    env_fns = [make_env_fn_eval(i, i + 420, config.test_word_file, config.GRID_SHAPE, config.env_max_steps, live_vlm_provider) for i in range(config.n_eval_envs)]
    eval_vec_env = DummyVecEnv(env_fns)
    eval_vec_env = VecTransposeImage(eval_vec_env)
    if isinstance(eval_vec_env.unwrapped, DummyVecEnv):
        for i in range(config.n_eval_envs): eval_vec_env.unwrapped.envs[i].set_eval_mode(True)
    else: eval_vec_env.env_method("set_eval_mode", True, indices=list(range(config.n_eval_envs)))

    custom_objects_for_load = {
        "learning_rate": 0.0003,
        "clip_range": 0.2,
        "__main__.VLMExtractor": VLMExtractor
    }

    print("Attempting to load PPO model...")
    loaded_model = PPO.load(
        config.model_path, 
        device=config.device, 
        env=eval_vec_env,
        custom_objects=custom_objects_for_load
    )
    print("PPO model loaded.")
    
    actual_extractor = None
    if hasattr(loaded_model.policy, 'features_extractor'):
        actual_extractor = loaded_model.policy.features_extractor
    
    if actual_extractor is not None and \
       getattr(actual_extractor.__class__, '__name__', None) == 'VLMExtractor':
        print("  SUCCESS: Loaded model's feature extractor IS a VLMExtractor (by class name).")
        
        print(f"    Actual_extractor.agent_for_embed (unpickled by SB3): {type(actual_extractor.agent_for_embed)}, ID: {id(actual_extractor.agent_for_embed)}")
        print(f"    Actual_extractor.proj (weights loaded by SB3): {type(actual_extractor.proj)}, ID: {id(actual_extractor.proj)}, Device: {actual_extractor.proj.weight.device}")

        print(f"    Reconfiguring the VLM components of the extractor's internal agent...")
        
        unpickled_agent_in_extractor = actual_extractor.agent_for_embed
        
        unpickled_agent_in_extractor.model = live_vlm_provider.model
        unpickled_agent_in_extractor.tokenizer = live_vlm_provider.tokenizer
        unpickled_agent_in_extractor.processor = live_vlm_provider.processor
        unpickled_agent_in_extractor.device = live_vlm_provider.device
        
        print(f"    Reconfiguration complete. Extractor will use:")
        print(f"        VLM components from live_vlm_provider (Model ID: {id(unpickled_agent_in_extractor.model)})")
        print(f"        Its own loaded & trained proj layer (ID: {id(actual_extractor.proj)}, Device: {actual_extractor.proj.weight.device})")
        if hasattr(actual_extractor, 'vlm_internal_batch_size'):
                    actual_extractor.vlm_internal_batch_size = config.vlm_internal_batch_size_eval

    else:
        print(f"  CRITICAL WARNING: Loaded model's feature extractor is NOT a VLMExtractor as expected.")
        print(f"     Actual extractor type: {type(actual_extractor)}")
        return 
    
    print("\n--- Post-load VLMExtractor Configuration Check ---")
    print(f"Extractor Type: {type(loaded_model.policy.features_extractor)}")
    if isinstance(loaded_model.policy.features_extractor, VLMExtractor):
        print(f"Extractor Agent Type: {type(loaded_model.policy.features_extractor.agent)}")
        print(f"Extractor Agent's VLM Model: {type(loaded_model.policy.features_extractor.agent.model)}")
        print(f"Extractor Agent's Proj (GOLKeyAgent's own): {type(loaded_model.policy.features_extractor.agent.proj)}")
        print(f"Extractor's Own Proj (used in forward): {type(loaded_model.policy.features_extractor.proj)}")
        print(f"Extractor's Proj Device: {loaded_model.policy.features_extractor.proj.weight.device}")
        print(f"Extractor Agent Device: {loaded_model.policy.features_extractor.agent.device}")
    print(f"PPO Policy Device: {loaded_model.policy.device}\n")

    if hasattr(loaded_model.policy.features_extractor, 'agent') and \
       isinstance(loaded_model.policy.features_extractor.agent, GOLKeyAgent):
        
        vlm_device_str = str(loaded_model.policy.features_extractor.agent.device)
        ppo_policy_device_str = str(loaded_model.policy.device)

        print(f"  INFO: VLM (GOLKeyAgent within extractor) is operating on device: {vlm_device_str}")
        print(f"  INFO: PPO policy networks are on device: {ppo_policy_device_str}")

        if torch.cuda.device_count() == 1:
            if vlm_device_str == "cuda:0": vlm_device_str = "cuda"
            if ppo_policy_device_str == "cuda:0": ppo_policy_device_str = "cuda"
            
        if vlm_device_str != ppo_policy_device_str:
            print(f"  WARNING: Device mismatch detected! VLM device ({loaded_model.policy.features_extractor.agent.device}) "
                  f"and PPO policy device ({loaded_model.policy.device}) may lead to data transfers.")
        else:
            print(f"  INFO: VLM and PPO policy devices appear consistent ({vlm_device_str}).")
    
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

    while len(results) < num_total_words_to_test:
        if num_active_envs == 0: break

        actions, _ = loaded_model.predict(obs, deterministic=True)
        next_obs, step_rewards, dones, infos = eval_vec_env.step(actions)

        for i in range(config.n_eval_envs):
            if not active_envs_mask[i]: continue

            current_episode_rewards[i] += step_rewards[i]
            current_episode_steps[i] += 1

            if dones[i]:
                num_completed_overall = len(results)
                processed_word = current_word_for_env[i] # Word for this episode
                episode_success = infos[i].get('success', False)
                episode_final_reward = current_episode_rewards[i] # Final reward for this episode
                episode_total_steps = current_episode_steps[i]    # Total steps for this episode
                episode_final_prefix = infos[i].get('prefix', 0)

                # Modified print statement
                if num_completed_overall < 10 or (num_completed_overall + 1) % 50 == 0 : # Log first 10, then every 50th
                    print(f"    Env {i}: Word '{processed_word}' | "
                          f"Success: {episode_success} | "
                          f"Steps: {episode_total_steps} | "
                          f"Reward: {episode_final_reward:.2f} | "
                          f"Prefix: {episode_final_prefix} | "
                          f"Total Results: {num_completed_overall + 1}/{num_total_words_to_test}")
                
                if processed_word: # Should always be true if active_envs_mask[i] was true
                    results.append({
                        'target_word': processed_word,
                        'word_len': len(processed_word),
                        'success': episode_success,
                        'steps': episode_total_steps,
                        'reward': episode_final_reward,
                        'final_prefix': episode_final_prefix
                    })
                    pbar.update(1)
                
                # Reset episodic trackers for this environment
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