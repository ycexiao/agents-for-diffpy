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

    @property
    @abstractmethod
    def parameter_names(self):
        """
        Return a sorted list of all parameter names.
        """
        pass

    @property
    @abstractmethod
    def parameters_slots(self):
        """
        Return a sorted list of all parameter names used in ML/RL model.
        The parameter name might not appear in the specific structure model.
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

    @abstractmethod
    def apply_parameters_values(self, pv_dict: dict):
        """
        Apply all parameter values from the provided dictionary.
        Raise KeyError if any parameter is missing.

        Parameters
        ----------
        pv_dict : dict
            Dictionary mapping parameter names to their desired values.
        """
        pass

    @abstractmethod
    def update_parameters_values(self, pv_dict: dict):
        """
        Only update given parameter values based on the provided dictionary to
        speed up the computing process.

        Parameters
        ----------
        pv_dict : dict
            Dictionary mapping parameter names to their desired values.
        """
        pass

    def get_parameters_values_dict(self):
        """
        Get a dictionary of current parameter values.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their current values.
        """
        return {pname: param.value for pname, param in self.parameters.items()}

    def apply_parameters_values(self, pv_dict: dict):
        """
        Apply all parameter values from the provided dictionary.
        Raise KeyError if any parameter is missing.
        """
        for pname, parameter in self.parameters.items():
            if pname not in pv_dict:
                continue
            parameter.setValue(pv_dict[pname])

    def update_parameters_values(self, pv_dict: dict):
        """
        Only update given parameter values based on the provided dictionary to
        speed up the computing process.
        """
        for pname, pvalue in pv_dict.items():
            if pname not in self.parameters:
                raise KeyError(f"Parameter {pname} not found in the model.")
            parameter = self.parameters[pname]
            parameter.setValue(pvalue)
