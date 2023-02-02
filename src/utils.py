from logging import Logger
import time


class CodeBlock:
    def __init__(self, logger: Logger, description: str):
        self.logger = logger
        self.description = description

    def __enter__(self):
        self.tic = time.time()
        self.logger.info(f"Entering {self.description} at {self.tic}")
        return

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        tac = time.time()
        self.logger.info(f"Exiting {self.description} at {tac}; wall-time = {tac-self.tic}")
        return