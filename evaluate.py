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

# --- Global reference for VLMExtractor dimensions during PPO.load ---
# This will be set to the live_vlm_provider before PPO.load
_live_vlm_for_extractor_dims_ref = None

class VLMExtractor(BaseFeaturesExtractor):
    """ Custom feature extractor using the GOLKeyAgent's embed method. """
    def __init__(self, observation_space, agent: "GOLKeyAgent", vlm_internal_batch_size: int):
        # 'agent' is the GOLKeyAgent instance unpickled by SB3 from policy_kwargs.
        # Its attributes (like patch_dim) might be unreliable if its internal VLM model
        # isn't fully initialized at this stage of PPO.load.
        global _live_vlm_for_extractor_dims_ref
        if _live_vlm_for_extractor_dims_ref is None:
            raise RuntimeError(
                "Global _live_vlm_for_extractor_dims_ref not set before VLMExtractor init by PPO.load. "
                "This reference is needed to ensure correct proj layer dimensions."
            )

        # Use dimensions from the live, fully initialized VLM provider
        # to ensure self.proj is created with the same dimensions as the trained layer.
        expected_intermediate_state = _live_vlm_for_extractor_dims_ref.intermediate_state
        expected_patch_dim = _live_vlm_for_extractor_dims_ref.patch_dim

        super().__init__(observation_space, features_dim=expected_intermediate_state)
        
        self.agent_for_embed = agent # Store the unpickled agent. Its VLM components will be replaced.
        
        # Create the projection layer. SB3 will load the trained weights into THIS layer.
        # Its dimensions MUST match the proj layer whose weights were saved during training.
        self.proj = torch.nn.Linear(expected_patch_dim, expected_intermediate_state)
        
        # DO NOT DO THIS: self.proj = agent.proj
        # The line above was the primary bug, causing the use of an untrained proj layer.

        self.vlm_internal_batch_size = vlm_internal_batch_size
        print(f"VLMExtractor Initialized for Eval: Features Dim = {self.features_dim}. "
              f"Unpickled agent for embed: {id(self.agent_for_embed)}. Proj layer (for SB3 to load weights into): {id(self.proj)}")
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # self.agent_for_embed will have its VLM components (model, tokenizer, processor)
        # replaced by those from the live_vlm_provider.
        with torch.no_grad():
            raw_features = self.agent_for_embed.embed(obs, max_batch=self.vlm_internal_batch_size)
        
        # self.proj is the layer created in __init__ with correct dimensions,
        # and SB3 has loaded the trained weights into it.
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
    print(f"Live GOLKeyAgent initialized. ID: {id(live_vlm_provider)}, Device: {live_vlm_provider.device}, Proj ID: {id(live_vlm_provider.proj)}")

    # Set the global reference for VLMExtractor dimensions BEFORE PPO.load
    global _live_vlm_for_extractor_dims_ref
    _live_vlm_for_extractor_dims_ref = live_vlm_provider

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
        "learning_rate": 0.0003, # Placeholder, actual value not used if policy loaded
        "clip_range": 0.2,       # Placeholder
        "__main__.VLMExtractor": VLMExtractor # Crucial: use the corrected VLMExtractor
    }

    print("Attempting to load PPO model...")
    loaded_model = PPO.load(
        config.model_path, 
        device=config.device, 
        env=eval_vec_env, # Provide env for observation/action space checks
        custom_objects=custom_objects_for_load
    )
    print("PPO model loaded.")

    # Clear the global reference after PPO.load has used it
    _live_vlm_for_extractor_dims_ref = None
    
    actual_extractor = None
    if hasattr(loaded_model.policy, 'features_extractor'):
        actual_extractor = loaded_model.policy.features_extractor
    else:
        print("CRITICAL: Loaded model's policy does not have a features_extractor attribute.")
        # Clean up global ref before returning
        _live_vlm_for_extractor_dims_ref = None
        return

    # More robust check for when the class is defined in __main__
    expected_class_name = "VLMExtractor"
    actual_class_name = getattr(actual_extractor.__class__, '__name__', None)

    if actual_extractor is not None and actual_class_name == expected_class_name:
        print(f"  SUCCESS: Loaded model's feature extractor IS a {expected_class_name} (by class name).")
         
        unpickled_agent_in_extractor = actual_extractor.agent
        
        print(f"    Actual_extractor's unpickled agent_for_embed: {type(unpickled_agent_in_extractor)}, ID: {id(unpickled_agent_in_extractor)}")
        print(f"    Actual_extractor's proj layer (weights loaded by SB3): {type(actual_extractor.proj)}, ID: {id(actual_extractor.proj)}, Device: {actual_extractor.proj.weight.device}")

        print(f"    Reconfiguring the VLM components of the extractor's internal agent_for_embed...")
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
            print("    Warning: actual_extractor does not have 'vlm_internal_batch_size'. Check VLMExtractor definition.")

    else:
        print(f"  CRITICAL WARNING: Loaded model's feature extractor is NOT a {expected_class_name} as expected.")
        print(f"     Actual extractor type: {type(actual_extractor)}")
        print(f"     Actual extractor class name: {actual_class_name}")
        # Clean up global ref before returning
        _live_vlm_for_extractor_dims_ref = None
        return 
        
    print("\n--- Post-load VLMExtractor Configuration Check ---")
    print(f"Extractor Type: {type(loaded_model.policy.features_extractor)}")
    if isinstance(loaded_model.policy.features_extractor, VLMExtractor):
        print(f"Extractor Agent_for_embed Type: {type(loaded_model.policy.features_extractor.agent_for_embed)}")
        print(f"Extractor Agent_for_embed's VLM Model: {type(loaded_model.policy.features_extractor.agent_for_embed.model)}")
        print(f"Extractor Agent_for_embed's Own Proj (GOLKeyAgent's internal, NOT USED by extractor.forward): {type(loaded_model.policy.features_extractor.agent_for_embed.proj)}")
        print(f"Extractor's Own Proj (used in VLMExtractor.forward): {type(loaded_model.policy.features_extractor.proj)}")
        print(f"Extractor's Proj Device: {loaded_model.policy.features_extractor.proj.weight.device}")
        print(f"Extractor Agent_for_embed Device: {loaded_model.policy.features_extractor.agent_for_embed.device}")
    print(f"PPO Policy Device: {loaded_model.policy.device}\n")

    if hasattr(loaded_model.policy.features_extractor, 'agent_for_embed') and \
       isinstance(loaded_model.policy.features_extractor.agent_for_embed, GOLKeyAgent):
        
        vlm_device_str = str(loaded_model.policy.features_extractor.agent_for_embed.device)
        ppo_policy_device_str = str(loaded_model.policy.device)

        print(f"  INFO: VLM (GOLKeyAgent within extractor, after VLM component replacement) is operating on device: {vlm_device_str}")
        print(f"  INFO: PPO policy networks are on device: {ppo_policy_device_str}")

        # Normalize "cuda:0" to "cuda" for comparison if single GPU
        if torch.cuda.device_count() == 1:
            if vlm_device_str == "cuda:0": vlm_device_str = "cuda"
            if ppo_policy_device_str == "cuda:0": ppo_policy_device_str = "cuda"
            
        if vlm_device_str != ppo_policy_device_str:
            print(f"  WARNING: Device mismatch detected! VLM device ({loaded_model.policy.features_extractor.agent_for_embed.device}) "
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
                processed_word = current_word_for_env[i] 
                episode_success = infos[i].get('success', False)
                episode_final_reward = current_episode_rewards[i] 
                episode_total_steps = current_episode_steps[i]    
                episode_final_prefix = infos[i].get('prefix', 0)
                
                if processed_word: 
                    results.append({
                        'target_word': processed_word,
                        'word_len': len(processed_word),
                        'success': episode_success,
                        'steps': episode_total_steps,
                        'reward': episode_final_reward,
                        'final_prefix': episode_final_prefix
                    })
                    pbar.update(1)
                
                # Log first few and then periodically
                if num_completed_overall < 5 or (num_completed_overall + 1) % 20 == 0 or episode_success:
                    print(f"    Env {i}: Word '{processed_word}' ({len(processed_word)}) | "
                          f"Success: {episode_success} | "
                          f"Steps: {episode_total_steps} | "
                          f"Reward: {episode_final_reward:.2f} | "
                          f"Prefix: {episode_final_prefix} | "
                          f"Total Results: {len(results)}/{num_total_words_to_test}")
                
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
            active_envs_needed = num_total_words_to_test - len(results)
            print(f"  Still need to test {active_envs_needed} words. Check logic if this persists.")
            # This might happen if n_eval_envs < num_total_words_to_test and some words take very few steps
            # The loop should continue as long as len(results) < num_total_words_to_test
            # and there are words to assign (next_word_idx < num_total_words_to_test)
            # The break condition `if num_active_envs == 0: break` is the one to watch.
            # It should be `if num_active_envs == 0 and next_word_idx >= num_total_words_to_test: break`
            # or rely on the outer `while len(results) < num_total_words_to_test:`
            if next_word_idx >= num_total_words_to_test: # All words have been dispatched at least once
                 break # No more words to assign, and all current ones are done.


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
        print("\n--- Performance by Word Length ---")
        grouped = results_df.groupby('word_len')
        summary_by_len = pd.DataFrame({
            'count': grouped.size(),
            'success_rate': grouped['success'].mean(),
            'mean_reward': grouped['reward'].mean(),
            'mean_steps': grouped['steps'].mean()
        })
        print(summary_by_len)


    results_df.to_csv(config.results_csv, index=False)
    print(f"\nEvaluation results saved to {config.results_csv}")
    print("--- Batched Evaluation Complete ---")


if __name__ == "__main__":
    eval_args = load_eval_config()
    evaluate_model_vec_batched(eval_args)