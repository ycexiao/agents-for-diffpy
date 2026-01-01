from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    @abstractmethod
    def load_inputs(self, inputs):
        pass

    @abstractmethod
    def apply_payload(self, payload):
        pass

    @abstractmethod
    def get_payload(self):
        pass

    @abstractmethod
    def action_func_factory(self, action_name):
        pass
