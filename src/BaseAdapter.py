from abc import ABC, abstractmethod


class BaseAdapter(ABC):

    @property
    @abstractmethod
    def residual(self):
        """
        Return the residual function to be minimized.
        """
        pass

    @property
    @abstractmethod
    def initial_values(self):
        """
        Return a dictionary of initial parameter values.
        """
        pass

    @property
    @abstractmethod
    def parameters(self):
        """
        Return a dictionary of parameter objects (both fixed and free).
        """
        pass

    @abstractmethod
    def free_parameters(self, parameter_names):
        """
        Free parameters given their names. Other parameters remain unchanged.

        Parameters
        ----------
        parameter_names : list of str
            List of parameter names to be freed.
        """
        pass

    @abstractmethod
    def fix_parameters(self, parameter_names):
        """
        Fix parameters given their names. Other parameters remain unchanged.

        Parameters
        ----------
        parameter_names : list of str
            List of parameter names to be fixed.
        """
        pass

    @abstractmethod
    def show_parameters(self):
        """
        Show current parameter values and their fix/free status.
        """
        pass
