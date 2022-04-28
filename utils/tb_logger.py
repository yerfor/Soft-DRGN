from tensorboardX import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.summary_writer = SummaryWriter(log_dir)

    def add_scalars(self, tag_step_value_dict):
        """
        :param parent_tag: str, e.g. "Training Loss"
        :param tag_step_value_dict: dict, e.g., {"key":(step, value), "q_grad":(10000, 1.11)}
        """
        for tag, (step, value) in tag_step_value_dict.items():
            self.summary_writer.add_scalar(tag, value, step)
