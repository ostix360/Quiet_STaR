class Quiet_STaR_config:

    def __init__(self,
                 thought_len=5,
                 n_thoughts=3,
                 n_truth=5,
                 start_thoughts_token_id=5,
                 end_thoughts_token_id=6,):
        self.thought_len = thought_len
        self.n_thoughts = n_thoughts
        self.n_truth = n_truth
        self.start_thoughts_token_id = start_thoughts_token_id
        self.end_thoughts_token_id = end_thoughts_token_id

