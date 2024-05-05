from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_plus_eps = list(vals)
    vals_plus_eps[arg] += epsilon

    vals_minus_eps = list(vals)
    vals_minus_eps[arg] -= epsilon

    return (f(*vals_plus_eps) - f(*vals_minus_eps)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    seen = set()
    sorted_variables = []

    def _topological_sort(variable: Variable) -> Iterable[Variable]:
        if variable.is_leaf() and variable.unique_id not in seen:
            seen.add(variable.unique_id)
            sorted_variables.append(variable)
        elif variable.is_constant():
            return
        else:
            if variable.unique_id not in seen:
                parents = variable.parents
                for parent in parents:
                    _topological_sort(parent)
                sorted_variables.append(variable)
                seen.add(variable.unique_id)

    _topological_sort(variable)
    top_sort = reversed(sorted_variables)
    return top_sort


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted = list(topological_sort(variable))

    print([(v.data, v.is_leaf()) for v in topological_sort(variable)])

    grads = {}
    grads[variable.unique_id] = deriv

    for var in sorted:
        if not var.is_leaf():
            grad_parents = var.chain_rule(grads[var.unique_id])
            for var_parent, grad_parent in grad_parents:
                if var_parent.unique_id not in grads.keys():
                    grads[var_parent.unique_id] = 0.0
                grads[var_parent.unique_id] += grad_parent

    for var in sorted:
        if var.is_leaf():
            var.accumulate_derivative(grads[var.unique_id])


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
