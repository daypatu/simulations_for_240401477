import argparse
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import time

from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csc_matrix
from pymatching import set_seed
from pymatching import Matching

class ToricMatching(nx.Graph):
    def __init__(self, size):
        super().__init__()
        self._size = size
        self.add_nodes_from(np.arange(size ** 2))
        for i in range(size ** 2):
            self.add_edge(i, (i // size) * size + (i + 1) % size)
            self.add_edge(i, (i + size) % size ** 2)
            self.edge_index_dict = {}
        for i, e in enumerate(self.edges()):
            self.edge_index_dict[e] = i
            self.edges[e[0], e[1]]["fault_ids"] = i
            self.x_membrane = np.zeros(self.number_of_edges(), dtype=int)
            self.y_membrane = np.zeros(self.number_of_edges(), dtype=int)
                                                                                                                                
        def fill_membrane(v, axis, membrane):
            for u in self.neighbors(v):
                point = self.id_to_pos(u)
                if point[axis] == 1:
                    membrane[self.edge_index_dict[tuple(sorted([v, u]))]] = 1
       
        for v in range(0, self._size ** 2, self._size):
            fill_membrane(v, 0, self.x_membrane)
        for v in range(self._size):
            fill_membrane(v, 1, self.y_membrane)

    def sample_erasures(self, p, n_samples):
        samples = np.random.rand(n_samples, self.number_of_edges()) < p
        return samples

    def sample_errors(self, p, n_samples):
        samples = np.random.rand(n_samples, self.number_of_edges()) < p 
        return samples

    def id_to_pos(self, id):
        return [id % self._size, id // self._size]

    def get_check_matrix(self):
        matrix = np.zeros(
            (self.number_of_nodes(), self.number_of_edges()), dtype=int
        )
        for i, e in enumerate(self.edges()):
            matrix[e[0], i] = 1
            matrix[e[1], i] = 1
        return matrix

    def plot(self, membrane=None, noise=None, weights=None):
        fig = plt.figure(figsize=(8,8))
        points = list(map(self.id_to_pos, range(self._size ** 2)))
        plt.scatter(*list(map(list, zip(*points))), color='black')

        for e_ind, edge in enumerate(self.edges()):
            points = list(map(self.id_to_pos, edge))
            for i in range(2):
                if abs(points[0][i] - points[1][i]) > 1:
                    if points[1][i] == 0:
                        points[1][i] = self._size
                    else:
                        points[0][i] = self._size
            
            color = 'blue'
            if membrane is not None:
                if self.__dict__[membrane + "_membrane"][e_ind] == 1:
                    color = 'green'
            if noise is not None:
                if noise[e_ind] == 1:
                    color = 'red'
            if weights is not None:
                if weights[e_ind] == 0:
                    color = 'white'
            plt.plot(*list(map(list, zip(*points))), color=color, linewidth=0.5)
            plt.axis("off")


def plot_cube(ax, size, color='blue', width=1.0):
    ax.plot([0,size-1,size-1,size-1,0,0,0],
            [0,0,0,size-1,size-1,size-1,0],
            [0,0,size-1,size-1,size-1,0,0],
            color=color, linewidth=width)
    ax.plot([0,0],[0,0],[size-1,0], color=color, linewidth=width)
    ax.plot([0,size-1],[0,0],[size-1,size-1], color=color, linewidth=width)
    ax.plot([0,0],[0,size-1],[size-1,size-1], color=color, linewidth=width)
    ax.plot([size-1,size-1],[size-1,size-1],[0,size-1], color=color, linewidth=width)
    ax.plot([size-1,0],[size-1,size-1],[0,0], color=color, linewidth=width)
    ax.plot([size-1,size-1],[size-1,0],[0,0], color=color, linewidth=width)


class CubicClusterBasedMatching(nx.Graph):
    def __init__(self, size, z_scale=1):
        super().__init__()
        assert size % 2 == 0
        self._size = size
        self._z_scale = z_scale
        self.add_nodes_from(np.arange(int(size ** 3 * z_scale)))
        self.add_edges()

        self.edge_index_dict = {}
        for i, e in enumerate(self.edges()):
            self.edge_index_dict[e] = i
            self.edges[e[0], e[1]]["fault_ids"] = i
        
        self.x_membrane = np.zeros(self.number_of_edges(), dtype=int)
        self.y_membrane = np.zeros(self.number_of_edges(), dtype=int)
        self.z_membrane = np.zeros(self.number_of_edges(), dtype=int)

        def fill_membrane(v, axis, membrane):
            for u in self.neighbors(v):
                point = self.id_to_pos(u)
                if point[axis] == 1:
                    membrane[self.edge_index_dict[tuple(sorted([v, u]))]] = 1
        
        for v in range(0, self._size ** 3, self._size):
            fill_membrane(v, 0, self.x_membrane)
        for start in range(self._size):
            for v in range(start, self._size ** 3, self._size  ** 2):
                fill_membrane(v, 1, self.y_membrane)
        for v in range(self._size ** 2):
            fill_membrane(v, 2, self.z_membrane)

        multiplicity = []
        for e in self.edges():
            multiplicity.append(self.get_edge_data(e[0], e[1])['multiplicity'])
        self.multiplicity = np.array(multiplicity)
    
    def add_edges(self):
        raise NotImplemented

    def id_to_pos(self, id):
        return [id % self._size, id % (self._size ** 2) // self._size, id // self._size ** 2]

    def get_check_matrix(self):
        matrix = np.zeros(
            (self.number_of_nodes(), self.number_of_edges()), dtype=int
        )
        for i, e in enumerate(self.edges()):
            matrix[e[0], i] = 1
            matrix[e[1], i] = 1
        return matrix

    def sample_erasures(self, p, n_samples):
        samples = (np.random.rand(n_samples, self.number_of_edges()) <
                1 - (1 - p) ** self.multiplicity)
        return samples

    def sample_errors(self, p, n_samples):
        samples = (np.random.rand(n_samples, self.number_of_edges()) <
                0.5 * (1 - (1 - 2 * p) ** self.multiplicity))
        return samples

    def set_weights(self, prob_error):
        for i, e in enumerate(self.edges()):
            p = 0.5 * (1 - (1 - 2  * prob_error) ** self.multiplicity[i])
            self.edges[e[0], e[1]]["weight"] = np.log((1 - p) / p)

    def plot(self, membrane=None, noise=None, syndrome=None, weights=None):
        fig = plt.figure(figsize=(16,12))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        points = list(map(self.id_to_pos, range(self._size ** 3)))
        color = 'black'
        s = 30
        if syndrome is not None:
            choose_color = lambda x: 'black' * (1 - x) + x * 'yellow'
            color = list(map(choose_color, syndrome))
            s = s + 12 * syndrome
            ax.scatter(*list(map(list, zip(*points))), color=color, s=s)

        plot_cube(ax, self._size, width=2.0)

        for e_ind, edge in enumerate(self.edges()):
            points = list(map(self.id_to_pos, edge))
            for i in range(3):
                if abs(points[0][i] - points[1][i]) > 1:
                    if points[1][i] == 0:
                        points[1][i] = self._size
                    else:
                        points[0][i] = self._size

            color = 'blue'
            width = 0.5
            if membrane is not None:
                if self.__dict__[membrane + "_membrane"][e_ind] == 1:
                    color = 'green'
                    width = 1.0
            if noise is not None:
                if noise[e_ind] == 1:
                    color = 'red'
                    width = 2.0
            if weights is not None:
                if weights[e_ind] == 0:
                    color = 'white'
            ax.plot(*list(map(list, zip(*points))), color=color, linewidth=width)
        
        plt.axis('off')


class BranchMatching(CubicClusterBasedMatching):
    def add_edges(self):
        size = self._size
        to_x = lambda v: (v // size) * size + (v + 1) % size
        to_y = lambda v: (v // size ** 2) * size ** 2 + (v + size) % size ** 2
        to_z = lambda v: (v + size ** 2) % size ** 3
        to_minus_x = lambda v: (v // size) * size + (v - 1) % size

        for i in range(size ** 3):
            self.add_edge(i, to_x(i), multiplicity=1)
            self.add_edge(i, to_y(i), multiplicity=1)
            x, y, _ = self.id_to_pos(i)
            if x % 2 + y % 2 == 1:
                self.add_edge(i, to_z(i), multiplicity=4)
                self.add_edge(i, to_y(to_x(i)), multiplicity=1)
                self.add_edge(i, to_y(to_minus_x(i)), multiplicity=1)

class Linear4Matching(CubicClusterBasedMatching):
    def add_edges(self):
        size = self._size
        to_x = lambda v: (v // size) * size + (v + 1) % size
        to_y = lambda v: (v // size ** 2) * size ** 2 + (v + size) % size ** 2
        to_z = lambda v: (v + size ** 2) % size ** 3
        to_minus_x = lambda v: (v // size) * size + (v - 1) % size
        to_minus_z = lambda v: (v - size ** 2) % size ** 3
        for i in range(size ** 3):
            self.add_edge(i, to_x(i), multiplicity=1)
            self.add_edge(i, to_y(i), multiplicity=1)
            x, y, _ = self.id_to_pos(i)
            if x % 2 + y % 2 == 1:
                self.add_edge(i, to_z(i), multiplicity=4)
                self.add_edge(i, to_y(to_x(i)), multiplicity=3)
                self.add_edge(i, to_y(to_minus_x(i)), multiplicity=3)
                if y % 2 == 0:
                    self.add_edge(i, to_y(to_x(to_z(i))), multiplicity=1)
                    self.add_edge(i, to_y(to_minus_x(to_z(i))), multiplicity=1)
                else:
                    self.add_edge(i, to_y(to_x(to_minus_z(i))), multiplicity=1)
                    self.add_edge(i, to_y(to_minus_x(to_minus_z(i))), multiplicity=1)

class Linear4ZMatching(CubicClusterBasedMatching):
    def add_edges(self):
        size = self._size
        size_z = int(self._size * self._z_scale)
        to_x = lambda v: (v // size) * size + (v + 1) % size
        to_y = lambda v: (v // size ** 2) * size ** 2 + (v + size) % size ** 2
        to_z = lambda v: (v + size ** 2) % (size ** 2 * size_z)
        to_minus_x = lambda v: (v // size) * size + (v - 1) % size
        to_minus_z = lambda v: (v - size ** 2) % (size ** 2 * size_z)
        for i in range(size ** 2 * size_z):
            self.add_edge(i, to_x(i), multiplicity=1)
            self.add_edge(i, to_y(i), multiplicity=1)
            x, y, _ = self.id_to_pos(i)
            if x % 2 + y % 2 == 1:
                self.add_edge(i, to_z(i), multiplicity=4)
                self.add_edge(i, to_y(to_x(i)), multiplicity=3)
                self.add_edge(i, to_y(to_minus_x(i)), multiplicity=3)
                if y % 2 == 0:
                    self.add_edge(i, to_y(to_x(to_z(i))), multiplicity=1)
                    self.add_edge(i, to_y(to_minus_x(to_z(i))), multiplicity=1)
                else:
                    self.add_edge(i, to_y(to_x(to_minus_z(i))), multiplicity=1)
                    self.add_edge(i, to_y(to_minus_x(to_minus_z(i))), multiplicity=1)

class FullCyclizationMatching(CubicClusterBasedMatching):
    def add_edges(self):
        size = self._size
        volume = size ** 3
        assert volume == len(self.nodes())
        self.add_nodes_from(len(self.nodes()) + np.arange(int(3 * size ** 3)))
        to_x = lambda v: (v // size) * size + (v + 1) % size
        to_y = lambda v: (v // size ** 2) * size ** 2 + (v + size) % size ** 2
        to_z = lambda v: (v + size ** 2) % size ** 3
        to_minus_x = lambda v: (v // size) * size + (v - 1) % size

        for i in range(size ** 3):
            self.add_edge(i, to_x(i), multiplicity=4)
            self.add_edge(i, to_y(i), multiplicity=4)
            self.add_edge(i, to_z(i), multiplicity=4)
            self.add_edge(i, volume + i, multiplicity=1)
            self.add_edge(i, volume + to_x(i), multiplicity=1)
            self.add_edge(i, volume + to_y(i), multiplicity=1)
            self.add_edge(i, volume + to_x(to_y(i)), multiplicity=1)
            self.add_edge(i, 2 * volume + i, multiplicity=1)
            self.add_edge(i, 2 * volume + to_x(i), multiplicity=1)
            self.add_edge(i, 2 * volume + to_z(i), multiplicity=1)
            self.add_edge(i, 2 * volume + to_x(to_z(i)), multiplicity=1)
            self.add_edge(i, 3 * volume + i, multiplicity=1)
            self.add_edge(i, 3 * volume + to_y(i), multiplicity=1)
            self.add_edge(i, 3 * volume + to_z(i), multiplicity=1)
            self.add_edge(i, 3 * volume + to_y(to_z(i)), multiplicity=1)


class StarMatching(CubicClusterBasedMatching):
    def add_edges(self):
        size = self._size
        to_x = lambda v: (v // size) * size + (v + 1) % size
        to_y = lambda v: (v // size ** 2) * size ** 2 + (v + size) % size ** 2
        to_z = lambda v: (v + size ** 2) % size ** 3
        to_minus_x = lambda v: (v // size) * size + (v - 1) % size

        for i in range(size ** 3):
            self.add_edge(i, to_x(i), multiplicity=4)
            self.add_edge(i, to_y(i), multiplicity=4)
            self.add_edge(i, to_z(i), multiplicity=4)

def update_data(filename, new_data):
    data = pd.concat([pd.read_csv(filename), new_data])
    data[data.columns[1:]] = data.groupby(['x',])[data.columns[1:]].transform('sum')
    data = data.drop_duplicates(subset=['x',])
    data.sort_values(by=['x'], inplace=True)
    data.to_csv(filename, index=False)

def simulate_point(
        syndrome_graph, prob_error, prob_erasure, membrane_ids, n_samples=1000):
    multiplicity = np.ones(syndrome_graph.number_of_edges(), dtype=int)
    if 'multiplicity' in syndrome_graph.__dict__:
        multiplicity = syndrome_graph.multiplicity
    start = time.time()
    erasure_shots = syndrome_graph.sample_erasures(prob_erasure, n_samples)
    error_shots = syndrome_graph.sample_errors(prob_error, n_samples)
    logging.debug("        erasure and error sampling time: {:2f}".format(time.time() - start))
    matching_graph = Matching(syndrome_graph)
    membranes = [
            syndrome_graph.__dict__[m_id + "_membrane"] for m_id in membrane_ids]
    start = time.time()
    success_events = np.prod([matching_graph.percolate_batch(erasure_shots, membrane)
        for membrane in membranes], axis=0).nonzero()
    logging.debug("        percolation checking time: {:.2f}".format(time.time() - start))
    if prob_error == 0:
        return success_events[0].shape[0]
    erasure_shots = erasure_shots[success_events]
    error_shots = error_shots[success_events] * (1 - erasure_shots)  # dangerous line!
    syndrome_shots = matching_graph.get_syndrome_batch(error_shots)
    start = time.time()
    weight_batch = np.log((1 + (1 - 2 * prob_error) ** multiplicity) /
                     (1 - (1 - 2 * prob_error) ** multiplicity)) * (1 - erasure_shots)
    logging.debug("        weights production time: {:2f}".format(time.time() - start))
    start = time.time()
    correction_batch = matching_graph.decode_batch_with_erasure(syndrome_shots, weight_batch)
    logging.debug("        decoding time: {:2f}".format(time.time() - start))
    cycle_batch = (correction_batch + error_shots) % 2
    successes = np.prod([(membrane * cycle_batch).sum(axis=1) % 2 == 0 for membrane in membranes], axis=0)
    return successes.sum()


def collect_data_for_threshold_curves(
        syndrome_graph_func, membrane_ids, c_error, c_erasure, x, n_samples=1000, sizes=[12, 16, 20]):
    assert len(sizes) == 3
    error_probs = c_error * x
    erasure_probs = c_erasure * x
    df = pd.DataFrame({'x': x,
                        'n{}'.format(sizes[0]): np.ones(x.size, dtype=int) * n_samples,
                        's{}'.format(sizes[0]): np.zeros(x.size, dtype=int),
                        'n{}'.format(sizes[1]): np.ones(x.size, dtype=int) * n_samples,
                        's{}'.format(sizes[1]): np.zeros(x.size, dtype=int),
                        'n{}'.format(sizes[2]): np.ones(x.size, dtype=int) * n_samples,
                        's{}'.format(sizes[2]): np.zeros(x.size, dtype=int)})
    for size in sizes:
        syndrome_graph = syndrome_graph_func(size) #, 1.5)
        logging.info("Graph of size = {} created".format(size))
        for i, (error_prob, erasure_prob) in enumerate(zip(error_probs, erasure_probs)):
            logging.info("    ers = {:.7}, err = {:.7}".format(erasure_prob, error_prob))
            n_successes = simulate_point(
                    syndrome_graph, error_prob, erasure_prob, membrane_ids, n_samples)
            df['s{}'.format(size)].values[i] = n_successes
    return df

def get_filename(syndrome_graph_func, membrane_ids, c_error, c_erasure, sizes):
    assert len(sizes) == 3
    class_name = str(syndrome_graph_func).split('.')[1].split("'")[0]
    c_name = ".ers{:.5}err{:.5}".format(c_erasure, c_error)
    sizes_name = ".S{}_{}_{}".format(sizes[0], sizes[1], sizes[2])
    membrane_name = ".M" + "_".join(membrane_ids)
    return class_name + sizes_name + membrane_name + c_name + '.csv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', metavar='id', type=int, nargs=1,
        help='model id')
    parser.add_argument(
        'sizes', metavar='S', type=int, nargs=3,
        help='syndrome graph dimensions')
    parser.add_argument(
        'membranes', metavar='M', type=str,
        nargs='+', help='membrane ids')
    parser.add_argument(
        'c_error', metavar='C_err', type=float, nargs=1,
        help='error coefficient')
    parser.add_argument(
        'c_erasure', metavar='C_ers', type=float, nargs=1,
        help='errasure coefficient')
    parser.add_argument(
        'range', metavar='R', type=float, nargs=3,
        help='range of variable x, in format [start, end, step]')
    parser.add_argument(
        'n_samples', metavar='N', type=int, nargs=1,
        help='number of samples')
    parser.add_argument('--debug', dest='logging_level', action='store_const',
                        const=logging.DEBUG, default=logging.INFO,
                        help='set logging level (default: INFO)')
    parser.add_argument('--clear_log', dest='clear_logfile', action='store_const',
            const=True, default=False, help='delete logfile before calculations')

    args = parser.parse_args()
    C_ERROR = args.c_error[0]
    C_ERASURE = args.c_erasure[0]
    MODELS = [ToricMatching, StarMatching, Linear4Matching, BranchMatching, FullCyclizationMatching]
    GRAPH = MODELS[args.model[0]]
    SIZES = args.sizes
    MEMBRANE_IDS = args.membranes
    N = args.n_samples[0]
    x = np.round(np.arange(*args.range), 7)
    
    filename = get_filename(GRAPH, MEMBRANE_IDS, C_ERROR, C_ERASURE, SIZES)
    logfilename = "logs/{}.log".format(filename)
    if args.clear_logfile and os.path.isfile(logfilename):
        os.remove("logs/{}.log".format(filename))
    logging.basicConfig(level=args.logging_level,
            format='%(asctime)s %(message)s', 
            handlers=[
                logging.FileHandler(logfilename),
                logging.StreamHandler()
            ])
    
    logging.info("Start calculations for range=({:.5}, {:.5}, {:.5}) and N={}".format(*args.range, N))
    df = collect_data_for_threshold_curves(GRAPH, MEMBRANE_IDS, C_ERROR, C_ERASURE, x, N, SIZES)

    logging.info("Save data")
    filename = "results/{}".format(filename)
    if os.path.exists(filename):
        update_data(filename, df)
    else:
        df.to_csv(filename, index=False)
