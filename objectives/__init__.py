from typing import Type

from objectives.base import Objective


def objective(objective, **kwargs) -> Type[Objective]:
    """Objective factory function"""
    if objective == "docking":
        from objectives.docking import DockingObjective

        return DockingObjective(**kwargs)
    if objective == "lookup":
        from objectives.lookup import LookupObjective

        return LookupObjective(**kwargs)

    raise NotImplementedError(f'Unrecognized objective: "{objective}"')
