import typing
from typing import get_type_hints, get_origin, get_args


def apply_overrides(cfg: typing.Any, overrides: typing.List[str]):
    """
    Applies overrides to a config object using its type hints for casting.
    """
    # Get all type hints from the configuration class once.
    hints = get_type_hints(type(cfg))

    for kv in overrides or []:
        key, val_str = kv.split("=", 1)

        # Get the specific type hint for the key being overridden.
        target_type = hints.get(key)
        processed_val = None

        # 1. Handle 'None' value
        if val_str.lower() in ("none", "null"):
            processed_val = None

        # 2. Handle boolean type
        elif target_type is bool:
            processed_val = val_str.lower() in ("true", "1")

        # 3. Handle list types (e.g., List[int], List[float])
        elif get_origin(target_type) in (list, typing.List):
            # Get the list's inner type (e.g., int from List[int])
            element_type = get_args(target_type)[0] if get_args(target_type) else str
            # Create the list by casting each comma-separated value
            processed_val = [element_type(v.strip()) for v in val_str.split(',')]

        # 4. Handle other simple types (int, float, str) if a hint exists
        elif target_type:
            processed_val = target_type(val_str)

        # 5. Fallback for untyped fields (or if casting fails)
        else:
            processed_val = val_str

        setattr(cfg, key, processed_val)
