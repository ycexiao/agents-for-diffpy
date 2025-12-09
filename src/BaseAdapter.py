from abc import ABC, abstractmethod


class BaseAdapter(ABC):

    @property
    def residual(self):
        """
        Return the residual function to be minimized.
        """
        pass

    @property
    @abstractmethod
    def residual_scalar(self):
        """
        Return the scalar residual value.
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

    @property
    @abstractmethod
    def parameter_values_in_slots(self):
        """
        Return a sorted list of all parameter names used in ML/RL model.
        The parameter name might not appear in the specific structure model.
        """
        pass

    @property
    @abstractmethod
    def parameter_names_in_slots(self):
        """
        Return a sorted list of all parameter names used in ML/RL model.
        The parameter name might not appear in the specific structure model.
        """
        pass

    @property
    @abstractmethod
    def parameter_slots_mask(self):
        """
        Return a boolean list indicating which parameters are free (True) or fixed (False).
        The order corresponds to `parameter_names_in_slots`.
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
    def apply_parameter_values_in_slot(self, values):
        """
        Apply parameter values from the `parameter_values_in_slots` to the
        current model.

        Parameters
        ----------
        values : list
            List of parameter values corresponding to `parameter_names_in_slots`.
        """
        pass
