

class ProcessingStep:
    """Base class for a step in the processing pipeline."""
    def __call__(self, data):
        raise NotImplementedError

class ProcessingPipeline:
    """
    Executes a sequence of processing steps on input data.
    """
    def __init__(self, *steps):
        self.steps = steps

    def run(self, initial_data):
        data = initial_data
        for step in self.steps:
            data = step(data)
        return data

