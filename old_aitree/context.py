

class Context:

    def __init__(self):
        self._blackboard = { }

    @property
    def blackboard(self):
        return self._blackboard

    def __contains__(self, key):
        return key in self._blackboard

    def __getitem__(self, key):
        return self._blackboard[key]

    def __setitem__(self, key, value):
        self._blackboard[key] = value
