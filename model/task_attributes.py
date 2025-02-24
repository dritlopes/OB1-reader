from dataclasses import dataclass

@dataclass
class TaskAttributes:
    """
    Class object that holds all parameters of task.

    Attributes:
        task_name (str): Name of the task.
        n_trials (int): Number of trials from the corpus/data should be used
        stim_cycles (str): The number of cycles per stimulus, if task has a set limit of time per trial.
        blank_screen_type (str): The type of blank screen in the experiment.
        blank_screen_cycles_begin (int): The number of model processing cycles at which the blank screen should appear.
        blank_screen_cycles_end (int): The number of model processing cycles at which the blank screen should end.
        is_priming_task (bool): Whether the task has a priming phase.
        n_cycles_prime_task (int): The number of model processing cycles priming phase.
    """
    task_name: str = 'reading'
    language: str = 'en'
    n_trials: int = 0,
    stim_cycles: int = 0,
    blank_screen_type: str = 'blank',
    blank_screen_cycles_begin: int = 0,
    blank_screen_cycles_end: int = 0,
    is_priming_task: bool = False,
    n_cycles_prime_task: int = 0
    affix_implemented: bool = False,
    POS_implemented = False

@dataclass
class EmbeddedWords(TaskAttributes):

    task_name: str = 'embedding_words'
    stim_cycles: int = 120 # each stimulus was on screen for 3s
    is_priming_task : bool = True,
    blank_screen_type: str ='hash_grid',
    blank_screen_cycles_begin : int = 5,  # blank screen before stimulus appears takes 200 ms  # FIXME : 20
    blank_screen_cycles_end : int = 0,
    n_cycles_prime : int = 2,  # prime was on the screen for 50ms
    affix_implemented: bool = True

@dataclass
class Flanker(TaskAttributes):

    task_name: str = 'flanker'
    stim_cycles: int = 7
    blank_screen_cycles_begin: int = 40,
    blank_screen_cycles_end: int = 40,
    affix_implemented: bool = True

@dataclass
class Transposed(TaskAttributes):

    task_name: str = 'transposed'
    stim_cycles: int = 120
    blank_screen_type : str = "fixation_cross"
    blank_screen_cycles_begin: int = 8,
    POS_implemented : bool = True,
    affix_implemented: bool = True