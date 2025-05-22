class PrivacyMeter:
    def __init__(self, epsilon: float, delta: float = None, accountant: str = 'rdp'):
        self.epsilon = epsilon
        self.delta = delta
        self.accountant = accountant
        self.usage = 0.0

    def consume(self, eps_increment: float, delta_increment: float = 0.0):
        self.usage += eps_increment
        if self.delta is not None:
            self.delta += delta_increment

    def is_exceeded(self):
        if self.delta is not None:
            return self.usage > self.epsilon or self.delta > self.delta
        return self.usage > self.epsilon

    def get_usage(self):
        return self.usage, self.delta
