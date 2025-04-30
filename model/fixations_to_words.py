import pandas as pd
from typing import Dict, List
from model_components import FixationOutput
import numpy as np

# Define a type alias for clarity
Millisecond = int

# https://link.springer.com/article/10.3758/s13428-017-0908-4
class FixationData:
    """Stores fixation-related eye-tracking data.

    Attributes:
        count (int): Total number of fixations in the interest area.
        first_start (Millisecond): Start time of the first fixation in the interest area.
        first_end (Millisecond): End time of the first fixation in the interest area.
    """

    def __init__(self, count: int, first_start: Millisecond, first_end: Millisecond):
        self.count = count
        self.first_start = first_start
        self.first_end = first_end

    @property
    def first_duration(self) -> Millisecond:
        """Computes the duration of the first fixation (end - start + 1)."""
        if self.first_start is None or self.first_end is None:
            return None
        return self.first_end - self.first_start + 1

    def to_provo_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using Provo dataset keys."""
        return {
            "IA_FIXATION_COUNT": self.count,
            "IA_FIRST_FIXATION_TIME": self.first_start,
            "IA_FIRST_FIXATION_DURATION": self.first_duration,
        }

    @staticmethod
    def from_provo_dict(data: Dict[str, int]):
        """Creates an instance from a dictionary."""
        return FixationData(
            count=data.get("IA_FIXATION_COUNT", 0),
            first_start=data.get("IA_FIRST_FIXATION_TIME", 0),
            first_end=data.get("IA_FIRST_FIXATION_TIME", 0) + data.get("IA_FIRST_FIXATION_DURATION", 1) - 1,
        )
    
    def to_meco_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using MECO dataset keys."""
        return {
            "nfix": self.count,
            "firstfix.dur": self.first_duration,
        }

    @staticmethod
    def from_meco_dict(data: Dict[str, int]):
        """Creates an instance from a MECO dictionary."""
        return FixationData(
            count=data.get("nfix", 0),
            first_start=0,                          # start time not specified in
            first_end=data.get("firstfix.dur", 0),
        )


class RunData:
    """Stores run-related eye-tracking data.

    Attributes:
        count (int): Number of times the interest area was entered and left.
        first_start (Millisecond): Start time of the first run of fixations in the interest area.
        first_end (Millisecond): End time of the first run of fixations in the interest area.
    """

    def __init__(self, count: int, first_start: Millisecond, first_end: Millisecond, total_time: Millisecond=0):
        self.count = count
        self.first_start = first_start
        self.first_end = first_end
        self.total_time = total_time

    @property
    def first_run_duration(self) -> Millisecond:
        """Computes the duration of the first run (end - start + 1)."""
        if self.first_start is None or self.first_end is None:
            return None
        return self.first_end - self.first_start + 1

    def to_provo_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using Provo dataset keys."""
        return {
            "IA_RUN_COUNT": self.count,
            "IA_FIRST_RUN_START_TIME": self.first_start,
            "IA_FIRST_RUN_END_TIME": self.first_end,
        }

    @staticmethod
    def from_provo_dict(data: Dict[str, int]):
        """Creates an instance from a dictionary."""
        return RunData(
            count=data.get("IA_RUN_COUNT", 0),
            first_start=data.get("IA_FIRST_RUN_START_TIME", 0),
            first_end=data.get("IA_FIRST_RUN_END_TIME", 0),
        )
    
    def to_meco_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using MECO dataset keys."""
        return {
            "nrun": self.count,
            "firstrun.dur": self.first_run_duration,
        }

    @staticmethod
    def from_meco_dict(data: Dict[str, int]):
        """Creates an instance from a MECO dictionary."""
        return RunData(
            count=data.get("nrun", 0),
            first_start=data.get("firstrun.nfix", 0),  # No exact equivalent for start time, using proxy
            first_end=data.get("firstrun.dur", 0),
        )


class SaccadeData:
    """Stores saccade-related eye-tracking data.

    Attributes:
        first_start (Millisecond): Start time of the first saccade landing in the area.
        first_end (Millisecond): End time of the first saccade landing in the area.
    """

    def __init__(self, first_start: Millisecond, first_end: Millisecond, first_amplitude: float):
        self.first_start = first_start
        self.first_end = first_end
        self.first_amplitude = first_amplitude

    @property
    def first_saccade_duration(self) -> Millisecond:
        """Computes the duration of the first saccade (end - start + 1)."""
        if self.first_start is None or self.first_end is None:
            return None
        return self.first_end - self.first_start + 1

    def to_provo_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using Provo dataset keys."""
        return {
            "IA_FIRST_SACCADE_START_TIME": self.first_start,
            "IA_FIRST_SACCADE_END_TIME": self.first_end,
            "IA_FIRST_SACCADE_AMPLITUDE": self.first_amplitude,
        }

    @staticmethod
    def from_provo_dict(data: Dict[str, int]):
        """Creates an instance from a dictionary."""
        return SaccadeData(
            first_start=data.get("IA_FIRST_SACCADE_START_TIME", 0),
            first_end=data.get("IA_FIRST_SACCADE_END_TIME", 0),
        )

    def to_meco_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using MECO dataset keys."""
        # data N/A: meco does not contain time information for saccade.     
        return {
        }

    @staticmethod
    def from_meco_dict(data: Dict[str, int]):
        """Creates an instance from a MECO dictionary."""
        # data N/A: meco does not contain time information for saccade.
        return SaccadeData(
            first_start=None,
            first_end=None,
            first_amplitude=None,
        )


class RegressionData:
    """Stores regression-related eye-tracking data.

    Attributes:
        in_count (int): Number of times the interest area was entered from a later part of the text.
        out_count (int): Number of times the interest area was exited to an earlier part of the text.
        out_full_count (int): Total number of regressions out of the interest area.
    """

    def __init__(self, in_count: int, out_count: int, out_full_count: int):
        self.in_count = in_count
        self.out_count = out_count
        self.out_full_count = out_full_count

    def to_provo_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using Provo dataset keys."""
        return {
            "IA_REGRESSION_IN_COUNT": self.in_count,
            "IA_REGRESSION_OUT_COUNT": self.out_count,
            "IA_REGRESSION_OUT_FULL_COUNT": self.out_full_count,
        }

    @staticmethod
    def from_provo_dict(data: Dict[str, int]):
        """Creates an instance from a dictionary."""
        return RegressionData(
            in_count=data.get("IA_REGRESSION_IN_COUNT", 0),
            out_count=data.get("IA_REGRESSION_OUT_COUNT", 0),
            out_full_count=data.get("IA_REGRESSION_OUT_FULL_COUNT", 0),
        )

    def to_meco_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using MECO dataset keys."""
        # 
        return {
            "reg.in": 1 if self.in_count>0 else 0,
            "reg.out": 1 if self.out_count>0 else 0,
        }

    @staticmethod
    def from_meco_dict(data: Dict[str, int]):
        """Creates an instance from a MECO dictionary."""
        return RegressionData(
            in_count=data.get("reg.in", 0),
            out_count=data.get("reg.out", 0),
            out_full_count=None, # N/A
        )


class WordLevelOutput:
    """Stores all eye-tracking data at the word level.

    Attributes:
        word (str): The word being processed.
        fixation (FixationData): Fixation-related data.
        run (RunData): Run-related data.
        saccade (SaccadeData): Saccade-related data.
        regression (RegressionData): Regression-related data.
    """

    def __init__(self, word: str, fixation: FixationData, run: RunData, saccade: SaccadeData, regression: RegressionData, skip_count: int, first_run_fixation_duration: int, fixation_duration: int):
        self.word = word
        self.fixation = fixation
        self.run = run
        self.saccade = saccade
        self.regression = regression
        self.skip_count = skip_count
        self.first_run_fixation_duration = first_run_fixation_duration
        self.fixation_duration = fixation_duration
    
    @property
    def skip(self):
        return 1 if self.skip_count>0 else 0

    def to_provo_dict(self) -> Dict[str, int]:
        """Converts the object to a dictionary using Provo dataset keys."""
        return {
            "Word": self.word,
            **self.fixation.to_provo_dict(),
            **self.run.to_provo_dict(),
            **self.saccade.to_provo_dict(),
            **self.regression.to_provo_dict(),
            "IA_SKIP": self.skip_count,
            "IA_FIRST_RUN_DWELL_TIME": self.first_run_fixation_duration,
            "IA_DWELL_TIME": self.fixation_duration
        }

    @staticmethod
    def from_provo_dict(data: Dict[str, int]):
        """Creates an instance from a dictionary."""
        return WordLevelOutput(
            word=data.get("Word"),
            fixation=FixationData.from_provo_dict(data),
            run=RunData.from_provo_dict(data),
            saccade=SaccadeData.from_provo_dict(data),
            regression=RegressionData.from_provo_dict(data),
            skip_count=data.get("IA_SKIP"),
            first_run_fixation_duration=data.get("IA_FIRST_RUN_DWELL_TIME"),
            fixation_duration=data.get("IA_DWELL_TIME")
        )
    
    def to_meco_dict(self) -> Dict[str, int]:
        return {
            "ia": self.word,
            **self.fixation.to_meco_dict(),
            **self.run.to_meco_dict(),
            **self.saccade.to_meco_dict(),
            **self.regression.to_meco_dict(),
            "skip": 1 if self.skip_count>0 else 0
        }
    
    @staticmethod
    def from_meco_dict(data: Dict[str, int]):
        return WordLevelOutput(
            word=data.get("ia"),
            fixation=FixationData.from_provo_dict(data),
            run=RunData.from_provo_dict(data),
            saccade=SaccadeData.from_provo_dict(data),
            regression=RegressionData.from_provo_dict(data),
            skip_count=data.get("skip")
        )

    @staticmethod
    def from_provo_dataframe(df: pd.DataFrame) -> List["WordLevelOutput"]:
        """Converts a DataFrame into a list of WordLevelOutput instances."""
        return [WordLevelOutput.from_provo_dict(row.to_dict()) for _, row in df.iterrows()]

    @staticmethod
    def to_provo_dataframe(objects: List["WordLevelOutput"]) -> pd.DataFrame:
        """Converts a list of WordLevelOutput instances into a Pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a word with associated eye-tracking metrics.
        """
        return pd.DataFrame([obj.to_provo_dict() for obj in objects])

class WordLevelStatistics:
    """
    Store statistics of word level outputs across different simulations
    """
    def __init__(self, outputs: list[WordLevelOutput]):
        self.word  = outputs[0].word
        self.mean_first_fixation_duration = np.mean([
            output.fixation.first_duration if output.fixation.first_duration is not None else 0
            for output in outputs
        ])
        self.mean_first_run_fixation_duration = np.mean([
            output.first_run_fixation_duration
            for output in outputs
        ])
        self.mean_total_fixation_duration = np.mean([
            output.fixation_duration for output in outputs
        ])
        self.mean_total_reading_time = np.mean([
            output.run.total_time if output.run.total_time else 0 for output in outputs
        ])
        self.skip_rate = np.mean([
            output.skip for output in outputs
        ])
        self.outgoing_regression_rate = np.mean([
            1 if output.regression.out_full_count else 0 for output in outputs
        ])
        self.incoming_regression_rate = np.mean([
            1 if output.regression.in_count else 0 for output in outputs
        ])

def fixation_to_word(words: list[str], fixations: list[FixationOutput]) -> list[WordLevelOutput]:
    """Converts a sequence of FixationOutput to a sequence of WordLevelOutput, aligning with the format in Provo."""
    
    outputs = [WordLevelOutput(
        word,
        FixationData(0,None,None,),
        RunData(0,None,None),
        SaccadeData(None,None,None),
        RegressionData(0,0,0),
        0,0,0
    ) for word in words]  # List to store WordLevelOutput instances

    n = len(fixations)

    current_max_focus = -1
    timestamp = 0
    for i in range(n):
        fixation = fixations[i]
        focus = fixation.fixation
        current_max_focus = max(current_max_focus, focus)
        assert words[focus]==fixation.fixated_word

        if outputs[focus].fixation.count == 0:
            outputs[focus].fixation.first_start = timestamp+1
            outputs[focus].run.first_start = timestamp+1

        timestamp += fixation.fixation_duration

        if outputs[focus].fixation.count == 0:
            outputs[focus].fixation.first_end = timestamp
        if outputs[focus].run.count == 0:
            outputs[focus].first_run_fixation_duration += fixation.fixation_duration
            outputs[focus].run.first_end = timestamp
        outputs[focus].fixation_duration += fixation.fixation_duration
        
        outputs[focus].fixation.count += 1
        outputs[focus].run.count += 1
        outputs[focus].run.total_time += fixation.fixation_duration

        if i==n-1:
            break # no need to deal with the incoming saccade for the last fixation

        next_fixation = fixations[i+1]
        next_focus = next_fixation.fixation

        if fixation.saccade_type == 'refixation':
            assert next_focus == focus
            outputs[focus].run.count -= 1   # revert the count: +1-1=0
        else:
            # TODO: check if refixation counts as one saccade affecting the saccade.first_start/first_end, or not

            # update saccade data
            if outputs[next_focus].saccade.first_start is None:
                outputs[next_focus].saccade.first_start = timestamp+1
                outputs[next_focus].saccade.first_amplitude = fixation.saccade_distance
            timestamp += abs(int(fixation.saccade_distance*10))
            # TODO: here we randomly assume a saccade speed of 10, but this may NOT be true.
            # we need to further read the paper and figure out how the saccade duration & speed is modelled.
            if outputs[next_focus].saccade.first_end is None:
                outputs[next_focus].saccade.first_end = timestamp

            # :param saccade_type: the type of incoming saccade: skipping, forward, regression or refixation.
            if fixation.saccade_type == 'wordskip':
                assert next_focus > focus+1
                for j in range(focus+1, next_focus):
                    outputs[j].skip_count += 1
            elif fixation.saccade_type == 'forward':
                assert next_focus == focus+1
                pass
            elif fixation.saccade_type == 'regression':
                assert next_focus < focus
                outputs[next_focus].regression.in_count += 1
                outputs[focus].regression.out_full_count += 1
                if current_max_focus == focus:
                    outputs[focus].regression.out_count += 1
            else:
                print(f"{fixation.saccade_type}")
                # raise NotImplementedError()

    # df = WordLevelOutput.to_provo_dataframe(outputs)

    return outputs
