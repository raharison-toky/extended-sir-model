import numpy as np
from matplotlib import pyplot as plt
import random
from string import ascii_letters
import copy
from sklearn import metrics

class graph:

    def __init__(self) -> None:
        self.nodes = {}
        self.vertices = {}

    def add_node(self,node):
        if isinstance(node,list):
            for i in node:
                self.nodes[node.name] = node
        else:
            self.nodes[node.name] = node

    def add_vertex(self,start,finish,const,coeffs,name=None):
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
                self.vertices[name] = vertex(start,finish,const,coeffs)

            else:
                s = "".join(random.choices(ascii_letters,k=10))
                while s in self.vertices.keys():
                    s = "".join(random.choices(ascii_letters,k=10))
                self.vertices[s] = vertex(start,finish,const,coeffs)

    def get_values(self,node):
        return np.array(self.nodes[node].get_values())


    def step(self):
        """"
        This is where all the calculations for Euler's method happen.
        Two dictionnaries with the initial values of each node is created.
        The first one is used to get initial values for the calculations, 
        the second one gets updated by the vertices to get the final values.
        After all the calculations have been made, the values in that dictionnary
        are appended to each node of the graph.
        """
        initial = {i.name:i.get_values()[-1] for i in self.nodes.values()}
        initial[None] = 0
        final = {i.name:i.get_values()[-1] for i in self.nodes.values()}
        final[None] = 0

        for i in self.vertices.values():
            # the c is the delta that corresponds to the vertex
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
        self.steps = steps
        for i in range(steps):
            self.step()

    def plot_all(self,dt):
        t = [dt * i for i in range(self.steps +1)]
        for i in self.nodes.values():
            plt.plot(t,i.get_values(),label=i.name)

class node:

    def __init__(self,name,tag,start=0) -> None:
        self.name = name
        self.tag = tag
        self.values = [start]

    def add_value(self,value):
        self.values.append(value)

    def get_values(self):
        return self.values

class vertex:

    def __init__(self,start,finish,const,coeffs) -> None:
        self.start = start
        self.finish = finish
        self.coeffs = coeffs
        self.const = const

class model:

    def __init__(self,initial_state) -> None:
        
        self.initial_state = copy.deepcopy(initial_state)
        self.parameters = self.initial_state.vertices

    def fit(self,target,target_node,n_epochs,indices = None,simulate_kwargs=None,start_deltas=None):

        new_values = self.parameters.copy()
        use_values = self.parameters.copy()

        if start_deltas is not None:
            deltas = start_deltas

        else:
            deltas = {i:j.const/10 for i,j in new_values.items()}


        m = copy.deepcopy(self.initial_state)
        m.simulate(*simulate_kwargs)

        if indices is not None:
            loss0 = metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node).flat[indices],squared=False)

        else:
            loss0 = metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node),squared=False)

        loss1 = loss0

        print(f"starting loss: {loss1}")
        
        for i in range(n_epochs):
            for j in new_values.keys():
                use_values[j].const += deltas[j]
                m = copy.deepcopy(self.initial_state)
                for key,value in use_values.items():
                    m.vertices[key] = value
                
                m.simulate(*simulate_kwargs)

                try:
                    if indices is not None:
                        l = metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node)[indices],squared=False)
                    else:
                        l = metrics.mean_squared_error(y_true=target,y_pred=m.get_values(target_node),squared=False)
                    if l < loss1:
                        new_values[j].const = use_values[j].const
                        loss1 = l

                    else:
                        deltas[j] = - deltas[j]*(l/loss0)
                
                except:
                    pass

            use_values = new_values.copy()
        
        print(f"final loss: {loss1}")
        return new_values
