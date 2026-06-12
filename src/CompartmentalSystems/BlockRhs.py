from typing import Callable, List, Tuple
from functools import reduce
import numpy as np

def prod(tup):
  return reduce(lambda acc,el: acc*el,tup)

class BlockRhs:
    def __init__(
        self,
        time_str: str,
        func_tups: List[Tuple[Callable, List[str]]],  # noqa
   ):
        """
        The first argument 'time_str' denotes the alias for the t argument to
        be used later in the signature of the block functions.
        """
        self.time_str = time_str
        self.func_tups = func_tups

    def flat_rhs(
        self,
        block_shapes: List[Tuple[str, np.ndarray]]
    ) -> Callable[[np.double, np.ndarray], np.ndarray]:
        """
        The function returns a function dot_X=f(t,X) suitable as the right-hand
        side for the ode solver scipy.solve_ivp from a collection of array
        valued functions that compute blocks of dot_X from time and blocks of X
        rather than from single equations.

        A special application is the creation of block triangular systems, to
        integrate variables whose time derivative depends on the solution
        of an original system instantaneously along with it.

        Assume that
        X_1(t) is the solution of the initial value problem (ivp)

        ivp_1:
        dot_X_1=f_1(t,X), X_1(t_0)   

        and X_2(t) the solution of another ivp

        ivp_2:
        dot_X_2=f_2(t,X_1,X_2), X_2(t_0) whose right-hand side depends on x_1.

        Then we can obtain the solution of both ivps simultaneously by
        combining them into one.

        (dot_X_1, dox_X_2)^t = (f_1(t,X_1),f_2(t,X_1,X_2))^t

        For n instead of 2 variables one has:
        (dot_X_1, dot_X_2,..., dot_X_n)^t
            = (f_1(t,X_1), f_2(t,X_1,X_2),..., f_n(t,X_1,...X_n))^t

        For a full lower triangular system the block derivative dot_X_i
        depends on t and ALL the blocks X_1,...,X_i
        but often it will only depend on
        SOME of the previous blocks so that f_i has a considerably
        smaller argument list.

        This function therefore allows to specify WHICH blocks the f_i depend
        on.
        Consider the following 5+2*2 = 9 -dimensional block diagonal example:

        b_s=flat_rhs(
             time_str='t'
            ,start_blocks=[('X1',np.ones((5,1)),('X2',np.ones((2,2)))]
            ,func_tups=[
                 ((lambda x   : x*2 ),  ['X1']    )
                ,((lambda t,x : t*x ),  ['t' ,'X2'])
             ])

        The second argument 'start_blocks' describes the decomposition of X
        into blocks by a list of tuples of the form ('Name',array).
        The third argument 'func_tups' is a list of tuples of the function
        itself and the list of the names of its block arguments as specified
        in the 'start_blocks' argument.
        Order is important for the 'start_blocks' and the 'func_tups'.
        It is assumed that the i-th function computes the derivative of the
        i-th block.
        The names of the blocks itself are arbitrary and have no meaning apart
        from their  correspondence in the start_blocks and func_tups argument.
        """
        shape_dict = {n: sh  for (n,sh)  in block_shapes}
        block_names = [t[0] for t in block_shapes]
        sizes = [prod(sh) for (n,sh) in block_shapes]
        nb = len(sizes)
        # first cotmpute the indices of block boundaries in X by summing the
        # dimensiorns of the blocks
        # indices =  np.array([0] + [sum(sizes[:(i+1)]) for i in range(nb)])
        indices = [0] + [sum(sizes[:(i+1)]) for i in range(nb)]
        # make sure that all argument lists are really lists
        func_tups = self.func_tups
        strArgLists = [f[1] for f in func_tups]
        assert(all([isinstance(l, list) for l in strArgLists]))
        # make sure that the function argument lists do not contain
        # block names that are not mentioned in the Xblocks argument
        flatArgList = [arg for argList in strArgLists for arg in argList]
        assert(set(flatArgList).issubset(block_names+[self.time_str]))


        def flat_rhs(t, X):
            vecBlockDict = {block_names[i]: X[indices[i]:indices[i+1]]
                            for i in range(nb)}
            blockDict = {
                name: vecBlock.reshape( shape_dict[name])
                for name, vecBlock in vecBlockDict.items()
            }
            blockDict[self.time_str] = t
            arg_lists = [
                [blockDict[name] for name in ft[1]]
                for ft in func_tups
            ]
            vecResults = [
                func_tups[i][0](*arg_lists[i]).flatten()
                for i in range(nb)
            ]
            return np.concatenate(vecResults)


        return flat_rhs
