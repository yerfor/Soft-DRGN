class BaseTrainer:
    def __init__(self):
        self.work_dir = None
        self.log_dir = None

        self.env = None
        self.agent = None
        self.replay_buffer = None
        self.optimizer = None
        self.tb_logger = None
        raise NotImplementedError

    def i_iter_dict(self):
        """
        log the number of update iteration of the network
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_from_checkpoint_if_possible(self):
        raise NotImplementedError

    def _interaction_step(self, log_vars):
        raise NotImplementedError

    def _training_step(self, log_vars):
        raise NotImplementedError

    def _testing_step(self, log_vars):
        raise NotImplementedError

    def run(self):
        """
        This is the main function of training.
        :return:
        """
        while True:
            self._interaction_step()
            self._training_step()
            self._testing_step()
