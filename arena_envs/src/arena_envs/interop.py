def environment_registration_callback() -> list[str]:
    import arena_envs.touch_sphere  # noqa: F401 — triggers @register_environment
    from isaaclab_arena.environments.isaaclab_interop import (
        environment_registration_callback as arena_callback,
    )
    return arena_callback()