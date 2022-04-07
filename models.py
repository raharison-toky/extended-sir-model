import numpy as np
from matplotlib import pyplot as plt

class graph:

    def __init__(self) -> None:
        self.nodes = {}
        self.vertices = []

    def add_node(self,node):
        if isinstance(node,list):
            for i in node:
                self.nodes[node.name] = node
        else:
            self.nodes[node.name] = node

    def add_vertex(self,start,finish,const,coeffs):
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
            self.vertices.append(vertex(start,finish,const,coeffs))

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

        for i in self.vertices:
            # the c is the delta that corresponds to the vertex
            c = i.const
            for j in i.coeffs:
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