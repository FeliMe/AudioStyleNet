import torch


class SolverEncoder:
    def __init__(self, config):
        super().__init__()

        # General
        self.save_dir = 'saves'
        self.config = config
        self.device = 'cuda' if (
            torch.cuda.is_available() and config.use_cuda) else 'cpu'
        self.global_step = 0
        self.t_start = 0
        print("Training on {}".format(self.device))

        # Models
        self.encoder = None
