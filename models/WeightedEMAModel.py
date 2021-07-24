class WeightEmaModel(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        current_state_dict = self.model.state_dict()
        for key in current_state_dict.keys():
            current_state_dict[key] = self.decay * current_state_dict[key] + (1 - self.decay) * new_state_dict[key]

        self.model.load_state_dict(current_state_dict)
