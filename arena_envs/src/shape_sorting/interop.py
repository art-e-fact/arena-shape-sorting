def environment_registration_callback() -> list[str]:
    import shape_sorting.shape_sorting_env  # noqa: F401 — triggers @register_environment

    # _install_build_registered_capture()
    # _install_gym_make_env_kwargs_forwarder()
    from isaaclab_arena.environments.isaaclab_interop import (
        environment_registration_callback as arena_callback,
    )
    return arena_callback()