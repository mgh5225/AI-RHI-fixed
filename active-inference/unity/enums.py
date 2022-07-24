import enum


class Condition(enum.Enum):
    """
    Enumeration used to configure the experimental condition of the environment (i.e. the position of the rubber arm)
    """
    Left = 0.0
    Center = 1.0
    Right = 2.0
    RandReachClose = 3.0
    RandReachFar = 4.0
    Break = 5.0


class VisibleArm(enum.Enum):
    """
    Enumeration used to configure the arm visibility of the environment's camera
    """
    RealArm = 0.0
    RubberArm = 1.0


class Stimulation(enum.Enum):
    """
    Enumeration used to configure the type of visuo-tactile stimulation the agent receives
    """
    Synchronous = 0.0
    Asynchronous = 1.0
