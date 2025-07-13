from enum import Enum

class ParameterServerStatus(Enum):
    """
    Enum for the status of the parameter server.
    """
    ACCEPTED = 0
    REJECTED = 1
    SHUTDOWN = 2