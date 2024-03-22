from abc import abstractmethod


class BaseSchedulerPerIter:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError


class BaseSchedulerPerEpoch:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError
