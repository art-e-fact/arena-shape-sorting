def environment_registration_callback() -> list[str]:
    import shape_sorting.shape_asset  # noqa: F401 — triggers @register_asset
    import shape_sorting.shape_sorting_env  # noqa: F401 — triggers @register_environment

    from isaaclab_arena.environments.isaaclab_interop import (
        environment_registration_callback as arena_callback,
    )
    return arena_callback()