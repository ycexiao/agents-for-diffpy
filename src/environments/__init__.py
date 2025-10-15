import gymnasium as gym

gym.envs.registration.register(
    id="pdf-single-phase", entry_point="environments.pdf_envs:SinglePhase"
)
