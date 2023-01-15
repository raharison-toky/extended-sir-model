from re import L
import numpy as np
from matplotlib import pyplot as plt
import random
from string import ascii_letters
import copy
from sklearn import metrics
import concurrent.futures
from operator import itemgetter

class graph:

    """
    Class for differential equation problems that may (or may not) be represented with a graph (like SIR models).
    Each node contains the evolution of a variable over time and the edges contain the change for a small dt.
    Any number of nodes and edges can be added before the simulation where every edge is computed.
    """

    def __init__(self) -> None:
        self.nodes = {}
        self.edges = {}

    def add_node(self,node):
        """
        Method that takes in a node and adds it to the graph's dictionnary of nodes
        The key for that node will be the name given to the node
        """
        if isinstance(node,list):
            for i in node:
                self.nodes[node.name] = node
        else:
            self.nodes[node.name] = node

    def add_edge(self,start,finish,const,coeffs,name=None):
        """
        Method that takes in the parameters of a edge to add a edge from these parameters to the graph's
        dictionnary of edges and return nothing.
        If a name is not given, a random string will be chosen as the key for that edge.
        If a starting node, an end node, or coefficients are missing from the graph's nodes, the function will return 0. 
        """
        names = [i.name for i in self.nodes.values()]
        if (start != None) and start not in names:
            print(f"{start} not there")
            return 0

        if (finish != None) and (finish not in names):
            print("finish not there")
            return 0

        for i in coeffs:
            if i not in names:
                print(f"{i} is not in coeffs")
                return 0

        else:
            if name is not None:
                self.edges[name] = edge(start,finish,const,coeffs)

            else:
                s = "".join(random.choices(ascii_letters,k=10))
                while s in self.edges.keys():
                    s = "".join(random.choices(ascii_letters,k=10))
                self.edges[s] = edge(start,finish,const,coeffs)

    def get_values(self,node):
        """
        Method that takes in the name of a node and returns the values of that node in a numpy array.
        """
        return np.array(self.nodes[node].get_values())

    def step(self):
        """"
        This is where all the calculations for Euler's method happen.
        Two dictionnaries with the initial values of each node is created.
        The first one is used to get initial values for the calculations, 
        the second one gets updated by the edges to get the final values.
        After all the calculations have been made, the values in that dictionnary
        are appended to each node of the graph.
        """
        initial = {i.name:i.get_values()[-1] for i in self.nodes.values()}
        initial[None] = 0
        final = {i.name:(i.get_values()[-1] if i.coeff == False else 0)  for i in self.nodes.values()}
        final[None] = 0

        for i in self.edges.values():
            # the c is the delta that corresponds to the edge
            c = i.const
            # print(f"c: {c}")
            for j in i.coeffs:
                # print(f"j: {j}")
                # print(f"initial[j]: {initial[j]}")
                c *= initial[j]

            final[i.start] -= c
            final[i.finish] += c

        for key,value in final.items():
            if key != None:
                self.nodes[key].add_value(value)

    def simulate(self,steps):
        """
        Method that takes in a number of steps and executes each edge at every step.
        """
        self.steps = steps
        for i in range(steps):
            self.step()

    def plot_all(self,dt,nodes=[]):
        """
        Method that takes in the value of dt and plots the values of all nodes over time.
        """
        if nodes == []:
            selected = self.nodes.values()
        else:
            selected = [self.nodes[i] for i in nodes]
        t = [dt * i for i in range(self.steps +1)]
        for i in selected:
            plt.plot(t,i.get_values(),label=i.name)

class node:

    """
    Class for a node that is meant to be added to a graph object.
    """

    def __init__(self,name,tag,start=0,coeff=False) -> None:
        """
        The init of a node requires a name, a tag, and an initial value.
        """
        self.name = name
        self.tag = tag
        self.values = [start]
        self.coeff = coeff

    def add_value(self,value):
        """
        Method for adding a value at the end of a node's list of values over time.
        """
        self.values.append(value)

    def get_values(self):
        return self.values

class edge:

    """
    Class for a edge that is meant to be added to a graph object.
    """

    def __init__(self,start,finish,const,coeffs) -> None:
        self.start = start
        self.finish = finish
        self.coeffs = coeffs
        self.const = const

class model:

    """
    Class for finding the coefficients of differential equations with Euler's method.
    """

    def __init__(self,initial_state) -> None:

        """
        The init requires a graph object that will be fit to a certain target.
        The initial_state will contain the starting values of each node and starting constants for edges.
        """
        
        self.initial_state = copy.deepcopy(initial_state)
        self.parameters = self.initial_state.edges

    def fit(self,targets,target_nodes,n_epochs,indices = None,simulate_kwargs=None,start_deltas=None,edges = []):
        """
        This method requires target values, the name of the node for target values, and a number of epochs.
        It first finds an initial loss between the prediction of the initial_state and the target.
        Then, it creates a copy of the initial_state for every epoch to change the constants of the edges using gradient descent.
        After each epoch, the edges of the best performing graph will be returned.
        If indices are given, the loss will be calculated with the predicted values at these indices.
        A simulate_kwargs can be given so that a child class of graph is given as an initial_state.
        A start_deltas can also be given to choose the starting step for gradient_descent.
        """
        new_values = self.parameters.copy()
        use_values = self.parameters.copy()

        if start_deltas is not None:
            deltas = start_deltas

        else:
            deltas = {i:j.const/10 for i,j in new_values.items()}

        m = copy.deepcopy(self.initial_state)
        m.simulate(*simulate_kwargs)
        loss0 = 0

        # to minimize bias towards certain targets, the mean squared error is mean-normalized
        averages = np.mean(targets,axis=1)

        for target,target_node,mean in zip(targets,target_nodes,averages):
            if indices is not None:
                #loss0 += 1-metrics.r2_score(y_true=target,y_pred=m.get_values(target_node).flat[indices])
                loss0 += metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node).flat[indices],squared=False)/mean

            else:
                #loss0 += 1-metrics.r2_score(y_true=target,y_pred=m.get_values(target_node))
                loss0 += metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node),squared=False)/mean

        loss1 = loss0

        print(f"starting loss: {loss1}")

        if edges == []:
            edges = new_values.keys()
        
        for i in range(n_epochs):
            for j in new_values.keys():
                use_values[j].const += deltas[j]
                m = copy.deepcopy(self.initial_state)
                for key,value in use_values.items():
                    m.edges[key] = value
                
                m.simulate(*simulate_kwargs)

                try:
                    l = 0
                    for target,target_node,mean in zip(targets,target_nodes,averages):
                        if indices is not None:
                            #l += 1-metrics.r2_score(y_true=target,y_pred=m.get_values(target_node).flat[indices])
                            l += metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node)[indices],squared=False)/mean
                        else:
                            #l += 1-metrics.r2_score(y_true=target,y_pred=m.get_values(target_node))
                            l += metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node),squared=False)/mean
                    if l < loss1:
                        new_values[j].const = use_values[j].const
                        loss1 = l

                    else:
                        deltas[j] = - deltas[j]*(l/loss0)
                
                except:
                    pass

            use_values = new_values.copy()
        
        print(f"final loss: {loss1}")
        return new_values,loss1

class general_case:

    """
    Class for finding the coefficients of differential equations from different starting points.
    This class tries to find the best local minimum by starting the gradient descent at different points.
    Multiprocessing reduces the time for that by running multiple optimizations at the same time.
    """

    def __init__(self,initial_state ,starting_points) -> None:
        """
        The init requires a graph object that will be fit to a certain target.
        The initial_state will contain multiple starting values all to optimize.
        """
        self.initial_state = copy.deepcopy(initial_state)
        self.starting_points = starting_points

    def _fit(self,targets,target_nodes,n_epochs,initial_parameters,indices = None,simulate_kwargs=None,start_deltas=None):
        """
        _fit is used to apply the fit method to each starting point
        """
        return model(initial_parameters).fit(targets,target_nodes,n_epochs,indices,simulate_kwargs,start_deltas)

    def fit(self,targets,target_nodes,n_epochs,indices = None,simulate_kwargs=None,start_deltas=None):
        """
        This method uses multiprocessing to simulteneously fit multiple starting points.
        It takes in the same arguments as the fit method for a model.
        It returns the best performing edges from all the starting points.
        """

        candidates = []
         
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self._fit,targets,target_nodes,n_epochs,i,indices,simulate_kwargs,start_deltas) for i in self.starting_points]

        for f in concurrent.futures.as_completed(results):
            candidates.append(f.result())

        return min(candidates,key=itemgetter(1))
