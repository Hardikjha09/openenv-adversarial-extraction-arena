"""
OpenEnv FastAPI server.
CRITICAL: max_concurrent_envs=16 prevents GRPO parallel rollout bottleneck.
This is the single most common failure mode — do not change this value.
"""
from openenv.core.env_server import create_fastapi_app
from env.extraction_env import AdversarialExtractionEnv, EnvAction, EnvObservation

def make_env():
    return AdversarialExtractionEnv(
        split="train",
        token_budget_mode="linear",
        perturbation_budget=200,
        adversary_policy="model"
    )

app = create_fastapi_app(
    env=make_env,
    action_cls=EnvAction,
    observation_cls=EnvObservation,
    max_concurrent_envs=16
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
