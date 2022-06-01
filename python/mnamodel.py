import os
import numpy as np
import networkx as nx
from pyvis.network import Network
import pandas as pd
from symengine import var, Matrix, Lambdify
from numba import njit

class MNAModel:
    """
    EXAMPLE OF USE USE:
    
    Let "eqns" and "endo_vars" be lists with equations and endogenous variables, respectively, stored as strings. E.g.
        eqns = ['x+y=A', 'x-y=B']
        endo_vars = ['x', 'y']
    
    A class instance called "model" is initialized by
    
        model = MNAModel(eqns, endo_vars)
    This reads in the equations and endogenous variables and perform block analysis and ordering and generates simulation code.
    The model is then ready to be solved subject to data (exogenous and initial values of endogenous variables) in a Pandas dataframe.
    
    Let "data" be a dataframe containing data on A and B and initial values for x and y. Then the model can be solved by
    
        solution = model.solve_model(data)
    
    Now "solution" is a Pandas dataframe with exactly the same dimensions as "data", but where the endogenous variables are replaced by the solutions to the model.
    The last solution is also stored in "model.last_solution".
    
    Somethin about dependecy graphs...
    """
    
    def __init__(self, eqns: list, endo_vars: list):
        """
        Reads in equations and endogenous variables and does a number of operations, e.g. analyzing block structure using graph theory.
        Stores a number of results in instance variables.
        
        Args:
            eqns_list (list): List of equations equations as strings
            endog_vars_list (list): List of endogenous variables as strings
        
        Returns:
            None.
        """
        
        self.__some_error = False
        self.__lag_notation = '___lag'
        self.__max_lag = 0
        self.__root_tolerance = 1e-10
        
        print('INITIALIZING MODEL...')
        
        # Model equations and endogenous variables are checked and stored as immutable tuples (as opposed to mutable lists)
        self.__eqns, self.__endo_vars = self.__initialize_model(eqns, endo_vars)
        
        print('ANALYZING MODEL...')
        
        # Analyzing equation strings to determine variables, lags and coefficients
        self.__eqns_with_details, self.__var_mapping = self.__analyze_equations()
        
        # Using graph theory to analyze equations using existing algorithms to establish minimum simultaneous blocks
        self.__equation_endo_var_bipartite_graph = self.__generate_equation_endo_var_bipartite_graph()
        self.__equation_endo_var_match = self.__find_maximum_bipartite_match()
        self.__model_digraph = self.__generate_model_digraph()
        self.__condenced_model_digraph, self.condenced_model_node_varlist_mapping = self.__generate_condenced_model_digraph()
        self.__augmented_condenced_model_digraph = self.__generate_augmented_condenced_model_digraph()
        
        # Generating everything needed to simulate model
        self.__simulation_code, self.__blocks = self.__generate_simulation_code()
    
    
    @property
    def eqns(self):
        return self.__eqns
    
    @property
    def endo_vars(self):
        return self.__endo_vars
    
    @property
    def max_lag(self):
        return self.__max_lag
    
    @property
    def blocks(self):
        block_mapping1 = {key: ''.join([val[0],'(', str(-val[1]), ')']) for key, val in self.__var_mapping.items() if val[1] > 0}
        block_mapping2 = {key: key for key, val in self.__var_mapping.items() if val[1] == 0}
        block_mapping = {**block_mapping1, **block_mapping2}
        return [[x[0], tuple([block_mapping[y] for y in x[1]]), x[2]] for x in self.__blocks]
    
    @property
    def root_tolerance(self):
        return self.__root_tolerance
    
    @property
    def last_solution(self):
        try:
            return self.__last_solution
        except AttributeError:
            print('ERROR: No solution exists')
    
    
    @root_tolerance.setter
    def root_tolerance(self, value):
        if type(value) != float:
            print('ERROR: tolerance for termination must be of type float')
            return
        if value <= 0:
            print('ERROR: tolerance for termination must be positive')
            return
        self.__root_tolerance = value
    
    
    def __initialize_model(self, eqns: list, endo_vars: list):
        """
        Imports lists containing equations and endogenous variables stored as strings.
        Checks that there are no blank lines, sets everything to lowercase and returns as tuples.
        
        Args:
            eqns (list): List of equations
            endo_vars (list): List of endogenous variables
        Returns:
            Tuples containing equations and endogenous variables as strings.
        """
        
        print('* Importing equations')
        for i, eqn in enumerate(eqns):
            if eqn.strip() == '':
                print('ERROR: There are blank lines in equation list')
                self.__some_error = True
                return None, None
            eqns[i]=eqns[i].lower()
        
        print('* Importing endogenous variables')
        for endo_var in endo_vars:
            if endo_var.strip() == '':
                print('ERROR: There are blank lines in endogenous variable list')
                self.__some_error = True
                return None, None
            endo_vars[i]=endo_vars[i].lower()
            
        return tuple(eqns), tuple(endo_vars)
    
    
    def __analyze_equations(self):
        """
        Returns equations and list of variables with and without lag notation.
        (-)-syntax is replaced with ___LAG_NOTATION.
        
        Returns:
            1) A list of equations and variables in equations with and without lag-notation.
            2) A mapping linking variables with (-)-notation to variable names and lags.
        """
        
        if self.__some_error:
            return None, None
        
        print('* Analyzing equation strings')
        
        eqns_with_and_without_lag_notation_and_var = []
        
        for eqn in self.__eqns:
            eqns_with_and_without_lag_notation_and_var.append([eqn, *self.__analyze_equation(eqn)])
            
        exog_var_mapping = {}
        for mapping in [x[3] for x in eqns_with_and_without_lag_notation_and_var]:
            exog_var_mapping.update(mapping)
        
        return tuple(eqns_with_and_without_lag_notation_and_var), exog_var_mapping


    def __analyze_equation(self, eqn: str):
        """
        Takes an equation string and parses it into numerics (special care is taken to deal with scientific notation), variables, lags and operators/brackets.
        I've written my own parser in stead of using some existing because it needs to take care of then (-)-notation for lags.
        
        Args:
            equation (str): String containing equation.
        Returns:
            1) An equation string with (-)-syntax replaced by LAG_NOTATION-syntax for lagged variables (e.g. 'x(-1)' --> 'xLAG_NOTATION1').
            2) A list of lists containing pairs of variables in the equation and variables in the equation with (-)-syntax replaced by LAG_NOTATION-syntax for lagged variables.
            3) A mapping linking variables with (-)-notation to variable names and lags.
        """
        
        if self.__some_error:
            return
        
        parsed_eqn_with_lag_notation, temp_vars, var_mapping = [], [], {}
        num, var, lag = '', '', ''
        is_num, is_var, is_lag, is_sci = False, False, False, False
        
        for chr in ''.join([eqn, ' ']):
            is_num = (chr.isnumeric() and not is_var) or is_num
            is_var = (chr.isalpha()  and not is_num) or is_var
            is_lag = (is_var and chr == '(') or is_lag
            is_sci = (is_num and chr == 'e') or is_sci
            
            # Check if character is something other than a numeric, variable or lag and write numeric or variable to parsed equation
            if chr in ['=','+','-','*','/','(',')',' '] and not (is_lag or is_sci):
                if is_num:
                    parsed_eqn_with_lag_notation.append(str(num))
                if is_var:
                    # Replace (-)-notation by LAG_NOTATION for lags and appends _ to the end to mark the end
                    pfx = '' if lag == '' else ''.join([self.__lag_notation, str(-int(lag.replace('(', '').replace(')', ''))), '_'])
                    parsed_eqn_with_lag_notation.append(''.join([var, pfx]))
                    temp_vars.append([''.join([var, lag]), ''.join([var, pfx])])
                    var_mapping[''.join([var, pfx])] = [var, 0 if lag == '' else -int(lag.replace('(', '').replace(')', ''))]
                    if lag != '':
                        self.__max_lag = max(self.__max_lag, -int(lag.replace('(', '').replace(')', '')))
                if chr != ' ':
                    parsed_eqn_with_lag_notation.append(chr)
                num, var, lag = '', '', ''
                is_num, is_var, is_lag = False, False, False
                continue
            
            if is_sci and chr.isnumeric():
                is_sci = False
            
            if is_num:
                num = ''.join([num, chr])
                continue
            
            if is_var and not is_lag:
                var = ''.join([var, chr])
                continue

            if is_var and is_lag:
                lag = ''.join([lag, chr])
                if chr == ')':
                    is_lag = False
        
        eqn_with_lag_notation=''.join(parsed_eqn_with_lag_notation)
        
        vars = []
        for var in temp_vars:
            if var not in vars:
                vars.append(var)

        return eqn_with_lag_notation, tuple(vars), var_mapping
    
    
    def __generate_equation_endo_var_bipartite_graph(self):
        """
        Generates bipartite graph connetcting equations (U) with endogenous variables (V).
        See https://en.wikipedia.org/wiki/Bipartite_graph for an explanation of what a bipartite graph is.
        
        Returns:
            Bipartite graph.
        """
        
        if self.__some_error:
            return
        
        print('* Generating bipartite graph connecting equations and endogenous variables')
        
        # Make nodes in bipartite graph with equations U (0) and endogenous variables in V (1)
        equation_endo_var_bipartite_graph = nx.Graph()
        equation_endo_var_bipartite_graph.add_nodes_from([i for i, _ in enumerate(self.__eqns)], bipartite=0)
        equation_endo_var_bipartite_graph.add_nodes_from(self.__endo_vars, bipartite=1)
        
        # Make edges between equations and endogenous variables
        for i, vars_in_eqn in enumerate([[x[0] for x in y[1]] for y in [[z[1], z[2]] for z in self.__eqns_with_details]]):
            for endo_var_in_eqn in [x for x in vars_in_eqn if x in self.__endo_vars]:
                equation_endo_var_bipartite_graph.add_edge(i, endo_var_in_eqn)
        
        return equation_endo_var_bipartite_graph
    
    
    def __find_maximum_bipartite_match(self):
        """
        Finds a maximum bipartite match (MBM) of bipartite graph connetcting equations (U) with endogenous variables (V).
        See https://www.geeksforgeeks.org/maximum-bipartite-matching/ for more on MBM.
        
        Returns:
            Dictionary with matches (both ways, i.e. U-->V and U-->U).
        """
        
        if self.__some_error:
            return
        
        print('* Finding maximum bipartite match (MBM) (i.e. associating every equation with exactly one endogenus variable)')
        
        # Use maximum bipartite matching to make a one to one mapping between equations and endogenous variables
        try:
            mamimum_bipartite_match = nx.bipartite.maximum_matching(self.__equation_endo_var_bipartite_graph, [i for i, _ in enumerate(self.__eqns)])
            if len(mamimum_bipartite_match)/2 < len(self.__eqns):
                self.__some_error = True
                print('ERROR: Model is over or under spesified')
                return
        except nx.AmbiguousSolution:
            self.__some_error = True
            print('ERROR: Unable to analyze model')
            return
        
        return mamimum_bipartite_match
    
    
    def __generate_model_digraph(self):
        """
        Makes a directed graph showing how endogenous variables affect every other endogenous variable.
        See https://en.wikipedia.org/wiki/Directed_graph for more about directed graphs.
        
        Returns:
            Directed graph showing endogenous variables network.
        """
        
        if self.__some_error:
            return
        
        print('* Generating directed graph (DiGraph) connecting endogenous variables using bipartite graph and MBM')
        
        # Make nodes in directed graph of endogenous variables
        model_digraph = nx.DiGraph()
        model_digraph.add_nodes_from(self.__endo_vars)
        
        # Make directed edges showing how endogenous variables affect every other endogenous variables using bipartite graph and MBM
        for edge in self.__equation_endo_var_bipartite_graph.edges():
            if edge[0] != self.__equation_endo_var_match[edge[1]]:
                model_digraph.add_edge(edge[1], self.__equation_endo_var_match[edge[0]])
        
        return model_digraph


    def __generate_condenced_model_digraph(self):
        """
        Makes a condencation of directed graph of endogenous variables. Each node of condencation contains strongly connected components; this corresponds to the simulataneous model blocks.
        See https://en.wikipedia.org/wiki/Strongly_connected_component for more about strongly connected components.
        
        Returns:
            1) Condencation of directed graph of endogenous variables
            2) Mapping from condencation graph node --> variable list
        """
        
        if self.__some_error:
            return
        
        print('* Finding condensation of DiGraph (i.e. finding minimum simulataneous equation blocks)')
        
        # Generate condensation graph of equation graph such that every node is a strong component of the equation graph
        condenced_model_digraph = nx.condensation(self.__model_digraph)
        
        # Make a dictionary that associate every node of condensation with a list of variables
        condenced_model_node_varlist_mapping = {}
        for node in tuple(condenced_model_digraph.nodes()):
            condenced_model_node_varlist_mapping[node] = tuple(condenced_model_digraph.nodes[node]['members'])
        
        return condenced_model_digraph, condenced_model_node_varlist_mapping
    
    
    def __generate_augmented_condenced_model_digraph(self):
        """
        Augments condencation graph with nodes and edges for exogenous variables in order to show what exogenous variables affect what strong components.
        
        Returns:
            Augmented condencation of directed graph of endogenous variables.
        """
        
        if self.__some_error:
            return
        
        augmented_condenced_equation_digraph = self.__condenced_model_digraph.copy()
        
        # Make edges between exogenous variables and strong components it is a part of
        for node in self.__condenced_model_digraph.nodes():
            for member in self.__condenced_model_digraph.nodes[node]['members']:
                for exog_var_adjacent_to_node in [x[0] for x in self.__eqns_with_details[self.__equation_endo_var_match[member]][2] if x[0] not in self.__endo_vars]:
                    augmented_condenced_equation_digraph.add_edge(exog_var_adjacent_to_node, node)
        
        return augmented_condenced_equation_digraph
    
    
    def __generate_simulation_code(self):
        """
        TBA
        """
        
        if self.__some_error:
            return
        
        print('* Generating simulation code (i.e. block-wise symbolic objective function, symbolic Jacobian matrix and lists of endogenous and exogenous variables)')
        
        simulation_code, blocks = [], []
        for node in reversed(tuple(self.__condenced_model_digraph.nodes())):
            block_endo_vars, block_eqns, block_eqns_with_lag_notation, block_exog_vars = [], [], [], set()
            for member in self.__condenced_model_digraph.nodes[node]['members']:
                i = self.__equation_endo_var_match[member]
                block_endo_vars.append(member)
                block_eqns.append(self.__eqns_with_details[i][0])
                block_eqns_with_lag_notation.append(self.__eqns_with_details[i][1])
                block_exog_vars.update([x[1] for x in self.__eqns_with_details[i][2]])
            block_exog_vars.difference_update(set(block_endo_vars))
            blocks.append(tuple([tuple(block_endo_vars), tuple(block_exog_vars), tuple(block_eqns)]))
            simulation_code.append(tuple([*self.__make_obj_func_and_jaco(tuple(block_eqns_with_lag_notation), tuple(block_endo_vars), tuple(block_exog_vars)),
                tuple(block_endo_vars), tuple(block_exog_vars), tuple(block_eqns_with_lag_notation)]))
        
        return tuple(simulation_code), tuple(blocks)
    
    
    @staticmethod
    def __make_obj_func_and_jaco(eqns: tuple, endo_vars: tuple, exog_vars: tuple):
        """
        TBA
        """
            
        endo_symb, exog_symb, func = [], [], []
        for endo_var in endo_vars:
            var(endo_var)
            endo_symb.append(eval(endo_var))
        for exog_var in exog_vars:
            var(exog_var)
            exog_symb.append(eval(exog_var))
        for eqn in eqns:
            lhs, rhs = eqn.split('=')
            func_row = eval('-'.join([''.join(['(', lhs.strip().strip('+').strip('-'), ')']), ''.join(['(', rhs.strip().strip('+').strip('-'), ')'])]))
            func.append(func_row)
        
        jaco = Matrix(func).jacobian(Matrix(endo_symb)).tolist()

        func_lambdify = Lambdify([*endo_symb, *exog_symb], func, cse=True)
        jaco_lambdify = Lambdify([*endo_symb, *exog_symb], jaco, cse=True)
        
        output_func = lambda val_list, *args: func_lambdify(*val_list, *args)
        output_jaco = lambda val_list, *args: jaco_lambdify(*val_list, *args)
        
        return output_func, output_jaco
    
    
    def solve_model(self, input_data: pd.DataFrame):
        """
        TBA
        """
        
        if self.__some_error:
            return
        
        print('SOLVING MODEL...')
        
        output_data_array = input_data.to_numpy(dtype=np.float64, copy=True)
        var_names = input_data.columns.str.lower().to_list()
        
        print('First period: {}, last period: {}'.format(input_data.index[self.__max_lag], input_data.index[output_data_array.shape[0]-1]))
        print('Solving', end=' ')
        
        for period in list(range(self.__max_lag, output_data_array.shape[0])):
            print(input_data.index[period], end=' ')
            for block, simulation_code in enumerate(self.__simulation_code):
                solution = self.__solve_system_of_equations(simulation_code[0],
                                                            simulation_code[1],
                                                            simulation_code[2],
                                                            simulation_code[3],
                                                            tuple([output_data_array, var_names]),
                                                            period)
                
                # If solution fails then print details about block and return
                if solution['status'] != 0:
                    print('\nERROR: Failed to solve block {}:'.format(block))
                    print(''.join(['Endogenous variables: ', ','.join(simulation_code[2])]))
                    print(','.join([str(x) for x in self.__make_list_of_endo_values(simulation_code[2], output_data_array, var_names, period)]))
                    print(''.join(['Exogenous variables: ', ','.join(simulation_code[3])]))
                    print(','.join([str(x) for x in self.__make_list_of_exog_values(simulation_code[3], output_data_array, var_names, period)]))
                if solution['status'] == 2:
                    return
                
                for i, endo_var in enumerate(simulation_code[2]):
                    output_data_array[period, var_names.index(endo_var)] = solution['x'][i]
        
        print('\nFinished')
        
        self.__last_solution = pd.DataFrame(output_data_array, columns=var_names, index=input_data.index)
        
        return self.__last_solution
    
    
    def __solve_system_of_equations(self, objective_function, jacobian_matrix, endo_var_list: tuple, exog_var_list: tuple, output_data_names: tuple, period: int):
        """
        TBA
        """
        output_data, variable_names = output_data_names
        solution = self.__newton_raphson(objective_function,
                    self.__make_list_of_endo_values(endo_var_list, output_data, variable_names, period),
                    args = self.__make_list_of_exog_values(exog_var_list, output_data, variable_names, period),
                    tol = self.__root_tolerance,
                    jac = jacobian_matrix)
        return solution
    
    
    def __make_list_of_exog_values(self, exog_vars: list, data: np.array, variable_names: list, period: int):
        """
        TBA
        """
        exog_var_vals = []
        for exog_var in exog_vars:
            exog_var_name, lag = self.__var_mapping[exog_var]
            exog_var_vals.append(self.__fetch_cell(data, period-lag, variable_names.index(exog_var_name)))
        return tuple(exog_var_vals)

    
    def __make_list_of_endo_values(self, endo_vars: list, data: np.array, variable_names: list, period: int):
        """
        TBA
        """
        endo_var_vals = []
        for endo_var in endo_vars:
            endo_var_vals.append(self.__fetch_cell(data, period, variable_names.index(endo_var)))
        return np.array(endo_var_vals, dtype=np.float64)
    

    @staticmethod
    @njit
    def __fetch_cell(array, row, col):
        return array[row, col]
    
    
    @staticmethod
    def __newton_raphson(f, init, **kwargs):
        """
        TBA
        """
        
        if 'args' in kwargs:
            args = kwargs['args']
        else:
            args = ()
        if 'jac' in kwargs:
            jac = kwargs['jac']
        else:
            print('ERROR: Newton-Raphson requires symbolic Jacobian matrix')
            return {'x': np.array(init), 'fun': np.array(f(init, *args)), 'success': False}
        if 'tol' in kwargs:
            tol = kwargs['tol']
        else:
            tol = 1e-10
        if 'maxiter' in kwargs:
            maxiter = kwargs['maxiter']
        else:
            maxiter = 10
        
        success = True
        status = 0
        x_i = init
        f_i = np.array(f(init.tolist(), *args))
        i = 0
        while np.max(np.abs(f_i)) > tol:
            if i == maxiter:
                success = False
                status = 1
                break
            try:
                x_i = x_i-np.matmul(np.linalg.inv(np.array(jac(x_i.tolist(), *args))), f_i)
            except np.linalg.LinAlgError:
                success = False
                status = 2
                break
            f_i = np.array(f(x_i, *args))
            i += 1
        
        return {'x': x_i, 'fun': f_i, 'success': success, 'status': status}
    
    
    def draw_blockwise_graph(self, variable: str, max_ancestor_generations: int, max_descentant_generations: int, in_notebook = False):
        """
        Draws a directed graph of block in which variable is along with max number of ancestors and descendants.
        Opens graph in browser.
        
        Args:
            variable (str): Variable who's block should be drawn
            max_ancestor_generations (int): Maximum number of anscestor blocks
            max_descentant_generations (int): Maximum number of descendant blocks
        
        Returns:
            None
        """
        
        if self.__some_error:
            return
        
        variable_node = [variable in self.__condenced_model_digraph.nodes[x]['members'] for x in self.__condenced_model_digraph.nodes()].index(True)
        
        ancr_nodes = nx.ancestors(self.__augmented_condenced_model_digraph, variable_node)
        desc_nodes = nx.descendants(self.__augmented_condenced_model_digraph, variable_node)
        
        max_ancr_nodes = {x for x in ancr_nodes if\
            nx.shortest_path_length(self.__augmented_condenced_model_digraph, x, variable_node) <= max_ancestor_generations}
        max_desc_nodes = {x for x in desc_nodes if\
            nx.shortest_path_length(self.__augmented_condenced_model_digraph, variable_node, x) <= max_descentant_generations}
        
        subgraph = self.__augmented_condenced_model_digraph.subgraph({variable_node}.union(max_ancr_nodes).union(max_desc_nodes))
        subgraph_to_pyvis = nx.DiGraph()
        
        print(''.join(['Graph of block containing {} with {} generations of ancestors and {} generations of decendants: '
                       .format(variable, max_ancestor_generations, max_descentant_generations), str(subgraph)]))

        # Loop over all nodes in subgraph (chosen variable, it's ancestors and decendants) and make nodes and edges in pyvis subgraph
        for node in subgraph.nodes():
            if node in self.__condenced_model_digraph:
                node_label = '\n'.join(self.condenced_model_node_varlist_mapping[node])\
                    if len(self.condenced_model_node_varlist_mapping[node]) < 10 else '***\nHUGE BLOCK\n***'
                node_title = '<br>'.join(self.condenced_model_node_varlist_mapping[node])
                if node == variable_node:
                    node_size = 200
                    node_color = 'gold'
                if node in max_ancr_nodes:
                    node_size = 100
                    node_color = 'green'
                if node in max_desc_nodes:
                    node_size = 100
                    node_color = 'teal'
            else:
                node_label = node
                node_title = node
                node_size = 100
                node_color = 'silver'
            subgraph_to_pyvis.add_node(node, label=node_label, title=node_title, shape='circle', size=node_size, color=node_color)
        
        subgraph_to_pyvis.add_edges_from(subgraph.edges())

        net = Network('2000px', '2000px', directed=True, notebook=in_notebook)
        net.from_nx(subgraph_to_pyvis)
        net.repulsion(node_distance=50, central_gravity=0.01, spring_length=100, spring_strength=0.02, damping=0.5)
        net.show('graph.html')
