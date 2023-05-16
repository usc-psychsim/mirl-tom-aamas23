# mirl-tom-aamas23
 
/environments/property_gridworld.py
- Dynamics of location property information
- Collaboration Dynamics
- Reward features for agent roles
- Add agent models

/features/propertyworld.py
- Not used, moved to property_gridworld.py


Environment Testing
/examples/property_world_trajectories.py
- Test dynamics
- Generate GT trajectory and .gif

Data Collection + Model Inference
/examples/property_world_trajectories_with_inference.py
- Generate GT trajectory with inference, saved to .pkl
- if want to change models, use add_agent_models() in property_gridworld.py

/examples/reward_model_multiagent_inference.py
- Add inference to the collected trajectories

/examples/load_trajs.ipynb
- generate plots

MIRL with ToM
/examples/multiagent_ToM_property_world_irl.py
- Line 68 of this code to switch learner agent
- Line 557-563 of trajectory.py to switch model distribution

Getting Stats of FC and Policy Divergence 
/examples/test_property_world_divergence.py
- EVALUATE_BY = 'EPISODES': get policy divergence
  + make sure to set the reward weights of "learner team" (line 125-148)
  + and the reward weights of "team" (line 77-100) should be the GT rewards
- EVALUATE_BY = 'FEATURES': get empirical and estimated FCs
- EVALUATE_BY = 'EMPIRICAL': get empirical FCs
  + make sure the reward weights of "team" (line 77-100) are the learned rewards
