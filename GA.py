import numpy as np
import warnings
import matplotlib.pyplot as plt
from itertools import chain
import operator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, Arrow, ArrowStyle
import math
import random
from colorama import Fore, Back, Style

global SqPosRot
global prev_min_distance
prev_min_distance = 0
global MinSqAreaGrid
global MaxSqAreaGrid
MinSqAreaGrid = 0
MaxSqAreaGrid = 0
global pinArray
global OrientedSq
OrientedSq = []
SqPosRot = []

NUM_OF_ELITE_CHROMOSOMES = 1
GRID_SEL_SIZE = 4
MUTATION_RATE = 0.01
orientations = 4  # 0,90,180,270


class Population:
    global SqPosRot
    def __init__(self, size, nets = None, component = None, minSqAreaGrid = None, maxSqAreaGrid = None ):
        global MinSqAreaGrid
        global MaxSqAreaGrid
        self._chromosomes = []
        MinSqAreaGrid =   minSqAreaGrid
        MaxSqAreaGrid =   maxSqAreaGrid
        self._comp = component
        self.set_chromosome(nets)




    def set_chromosome(self, net):
        global P1xrange
        global P2xrange

        SqPosRot = self._comp.getPartsPosition()

        pinarray = self._comp.getParts()
        # pinarray = np.copy(pinArray)
        i = 0
        # print("net", net)

        P1xrange = []
        P2xrange = []

        Sq1, P1 = net[i]
        Sq2, P2 = net[i + 1]

        Sq1 = self._comp.getPartsNames().index(Sq1)
        Sq2 = self._comp.getPartsNames().index(Sq2)
        warnings.simplefilter(action='ignore', category=FutureWarning)
        P1 = np.array([int(P1)])
        P2 = np.array([int(P2)])

        # print("Square no",Sq1+1,":", pinArray[Sq1],"Pin no: ", P1)
        # print("Square no",Sq2+1,":", pinArray[Sq2],"Pin no: ", P2)
        # print("Index of pin1 in Square1", np.argwhere(pinArray[Sq1] == P1))
        # print("Index of pin2 in Square2", np.argwhere(pinArray[Sq2] == P2))

        # rotate P1,P2 in 90,180,270 and store it's index position is respective squares for that rotation
        for x in range(4):
            s1 = np.rot90(pinarray[Sq1].getMatrix(), x)
            s2 = np.rot90(pinarray[Sq2].getMatrix(), x)
            if (((Sq1+1, x, s1.shape )) not in OrientedSq) :
                OrientedSq.append((Sq1+1, x, s1.shape ) )
            if (((Sq2+1, x,  s2.shape)) not in OrientedSq) :
                OrientedSq.append((Sq2+1, x, s2.shape ) )
            idx1 = np.argwhere(P1 == s1)
            idx2 = np.argwhere(P2 == s2)
            # print("rot",x,"pin", P2, np.rot90(pinArray[Sq2],x))
            if not idx1.size or not idx2.size:
                if not idx1.size:
                    print(Fore.BLUE + "Warning! Check net points or Part Size. In Matrix Part '",net[i][0],"' doesnot contain Pin '",net[i][1],"'")
                else:
                    print(Fore.BLUE + "Warning! Check net points or Part Size. In Matrix Part '",net[i+1][0],"' doesnot contain Pin '",net[i+1][1],"'")
                break
            P1xrange.append((idx1[0], x))
            P2xrange.append((idx2[0], x))

        P1xrange = np.array(P1xrange)
        P2xrange = np.array(P2xrange)
        # print(P2xrange, P2xrange[random.randrange(len(P2xrange))])
        # print(P1xrange, P1xrange[random.randrange(len(P1xrange))])

        for Pi_x in range(len(P1xrange)):
            for Pj_x in range(len(P2xrange)):
                P1x = P1xrange[Pi_x][0][0]
                P1y = P1xrange[Pi_x][0][1]
                P2x = P2xrange[Pj_x][0][0]
                P2y = P2xrange[Pj_x][0][1]
                c = Chromosome()
                c.set_net(net)
                c.set_genes([P1x, P1y, P2x, P2y, Pi_x, Pj_x])
                c.set_distbtwsqs(Sq1+1,Sq2+1, self._comp.getPartsMinDist()[Sq2-Sq1-1], self._comp.getPartsMaxDist()[Sq2-Sq1-1])
                c.calculate_data(Sq1,Sq2, SqPosRot)
                self._chromosomes.append(c)


    def get_component(self):
        return  self._comp

    def get_chromosomes(self):
        return  self._chromosomes

    def _print_population(self, gen_number = None, i = None):
        print("\n--------------------------------")
        print("Generation #", gen_number, "| Fittest chromosome fitness :", self.get_chromosomes()[0].get_fitness())
        # self.get_chromosomes()[0].plot(self.get_component().getParts(),i)
        print("TARGET #", "                            | Fitness: 5 ", "| PinDist: range(",
              self.get_chromosomes()[0].get_minDistBtwPins(), " - ", self.get_chromosomes()[0].get_maxDistBtwPins(), ")",
              "| GridArea: range(", MinSqAreaGrid, " - ", MaxSqAreaGrid, ")")

        i = 0
        for x in self.get_chromosomes():
            print("Chromosome #", i, " :", x, "| Fitness: ", x.get_fitness(), "| PinDist: ", x.get_pinDist(),
                  "           | GridArea: ", x.get_gridArea(), )
            i += 1


class Chromosome():
    global SqPosRot

    def __init__(self):
        self._genes   = []
        self._fitness = 0
        self._net     = []
        self._minDistBtwPins = 0
        self._maxDistBtwPins = 0
        self._gridArea = 0
        self._distBtwPins = 0
        self.sq1Starts = (0,0)
        self.sq1_pos = (0,0)
        self.sq2Starts = (0,0)
        self.sq2_pos = (0,0)
        self.sq1_dim = (0,0)
        self.sq2_dim = (0,0)

    def set_genes(self, genes = None):
            self._genes = genes

    def get_genes(self):
        return self._genes

    def set_net(self,net):
        self._net = net

    def get_net(self):
        return self._net

    def get_fitness(self):
        global prev_min_distance

        if prev_min_distance == 0:
            prev_min_distance = self._distBtwPins
        if not self._fitness:
            if self._distBtwPins < prev_min_distance :
                prev_min_distance = self._distBtwPins
                self._fitness += 1

            if self._minDistBtwPins <= self._distBtwPins <= self._maxDistBtwPins:
                self._fitness += 2

            if MinSqAreaGrid <= self._gridArea <= MaxSqAreaGrid:
                self._fitness += 2

        return self._fitness

    def plot(self, Parts = None, i = None):

        if self._fitness >= 4:

            d = self.get_genes()[0][0:3]
            e = self.get_genes()[0][3:]
            s1Orient = self.get_genes()[0][4]
            s2Orient = self.get_genes()[0][5]

            fig = plt.figure(figsize=[8, 8])
            ax = fig.add_subplot(111)
            ax2 = fig.add_subplot(111)
            # ax.grid(True)

            axins = inset_axes(ax, width="100%", height="100%",
                               bbox_to_anchor=(0,self.sq1_pos[1]),
                               bbox_transform=ax.transAxes, loc=3)


            r = Rectangle(self.sq1Starts, self.sq1_dim[0], self.sq1_dim[1], facecolor="lightgrey", ec="black", label="Sq1")
            ax.add_artist(r)
            pinArray1 = Parts[i].getMatrix()
            pinArray2 = Parts[i+1].getMatrix()
            Sq1Orient = np.rot90(pinArray1[self._net[0][0]-1], self.get_genes()[0][4])
            Sq2Orient = np.rot90(pinArray2[self._net[1][0]-1], self.get_genes()[0][5])
            for i, rowpins in enumerate(Sq1Orient):
                for j, pin in enumerate(rowpins):
                    if pin != -1:
                        r = Rectangle((i,j), 0.7, 0.7,facecolor="blue", ec="black",url=pin)
                        if pin == self._net[0][1]:
                            lineS = (i,j)
                            ax.annotate(pin, (i + 0.2, j + 0.2), color='r', weight='bold', fontsize=12, ha='center',
                                        va='center')

                        else:
                            ax.annotate(pin, (i+0.2,j+0.2), color='w', weight='bold',fontsize=12, ha='center', va='center')
                        ax.add_artist(r)


            r = Rectangle( self.sq2_pos, self.sq2_dim[0], self.sq2_dim[1], facecolor="lightgrey", ec="black" )
            ax.add_artist(r)
            for i, rowpins in enumerate(Sq2Orient):
                for j, pin in enumerate(rowpins):
                    if pin != -1:
                        r = Rectangle((self.sq2_pos[0] + i + 0.1, self.sq2_pos[1] + j + 0.1), 0.7, 0.7, facecolor="black", ec="black")

                        if pin == self._net[1][1]:
                            lineE = (self.sq2_pos[0] +i, self.sq2_pos[1] + j)

                            ax.annotate(pin, (self.sq2_pos[0] + i + 0.3, self.sq2_pos[1] + j + 0.3), color='r',
                                        weight='bold', fontsize=12, ha='center', va='center')

                        else:
                            ax.annotate(pin, (self.sq2_pos[0] +i + 0.3,self.sq2_pos[1] + j + 0.3), color='w', weight='bold', fontsize=12, ha='center', va='center')
                        ax.add_artist(r)
            # ax.text()
            P1x,P1y,P2x,P2y,O1,O2 = self.get_genes()[0]
            # l = Arrow(lineS[0],lineS[1],lineE[0],lineE[1],width=0.1, color='r')
            ax.annotate("Opt distance between pins = "+ str(self._distBtwPins),((lineS[0]+lineE[0])/2,(lineS[1]+lineE[1])/2), color='b', weight='bold',fontsize=12, ha='center', va='center')
            #
            # ax.add_artist(l)

            ax.axis([0, 20, 0, 20])
            # # Turn ticklabels of insets off
            for axi in [axins]:
                axi.tick_params(labelleft=False, labelbottom=False)

            plt.show()


    def calculate_data(self, Sq1, Sq2, SqPosRot):

        Pin_dist = round(math.sqrt((2 ** abs(self._genes[2] - self._genes[0] ) ) + (2 ** abs(self._genes[3] - self._genes[1] )) ), 2)
        self._distBtwPins = Pin_dist

        for sq_dim in OrientedSq:
            if (sq_dim[0],sq_dim[1]) == (self._net[0][0], self._genes[4]):
                self.sq1_dim = sq_dim[2]
                break
        for sq_dim in OrientedSq:
            if (sq_dim[0],sq_dim[1]) == (self._net[1][0], self._genes[5]):
                self.sq2_dim = sq_dim[2]
                break


        self.sq1Starts = (SqPosRot[Sq1][0],SqPosRot[Sq1][1])
        self.sq2Starts = (SqPosRot[Sq2][0],SqPosRot[Sq2][1])

        self.sq1_pos =  tuple(map(operator.add, self.sq1_dim, self.sq1Starts[::-1]))
        self.sq2_pos =  tuple(map(operator.add, self.sq2_dim, self.sq2Starts[::-1]))
        gridArea = max( self.sq1_pos[0],self.sq2_pos[0],self._minDistBtwPins) + max( self.sq1_pos[1],self.sq2_pos[1],self._minDistBtwPins)
        self._gridArea = gridArea

        self.get_fitness()


    def set_distbtwsqs(self,Sq1,Sq2,parts_min_distance, parts_max_distance):

        self._minDistBtwPins = parts_min_distance
        self._maxDistBtwPins = parts_max_distance

    def get_minDistBtwPins(self):
        return  self._minDistBtwPins

    def get_maxDistBtwPins(self):
        return  self._maxDistBtwPins

    def get_pinDist(self):
        return  self._distBtwPins

    def get_gridArea(self):
        return self._gridArea

    def __str__(self):
        return self._genes.__str__()


class GeneticAlgorithm():

    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))

    @staticmethod
    def _crossover_population(pop):

        randomnet = pop.get_chromosomes()[0].get_net()
        POPULATION_SIZE = len(max(randomnet, key=len))
        random.shuffle(randomnet)
        # crossover_pop = Population(0, randomnet[0])
        crossover_pop =  Population(0,randomnet[0:2],pop.get_component(), MinSqAreaGrid, MaxSqAreaGrid)
        for i in range(NUM_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = NUM_OF_ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm._select_block_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm._select_block_population(pop).get_chromosomes()[1]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
            i += 1

        return crossover_pop

    @staticmethod
    def _mutate_population(pop):
        randomnet = pop.get_chromosomes()[0].get_net()
        POPULATION_SIZE = len(max(randomnet, key=len))
        for i in range(POPULATION_SIZE):
            GeneticAlgorithm._mutate_chromosome(pop.get_chromosomes()[i])
        return pop

    @staticmethod
    def _crossover_chromosomes(chromosome1, chromosome2):

        crossover_chrom = Chromosome()
        crossover_chrom.set_genes([chromosome1.get_genes()[0:3] + chromosome1.get_genes()[3:]])
        #
        return crossover_chrom

    @staticmethod
    def _mutate_chromosome(chromosome):
        if random.random() < MUTATION_RATE:
            P1x = P1xrange[random.randrange(len(P1xrange))][0]
            P1y = P1xrange[random.randrange(len(P1xrange))][0]
            P2x = P2xrange[random.randrange(len(P2xrange))][0]
            P2y = P2xrange[random.randrange(len(P2xrange))][1]
            chromosome.set_genes([P1x, P1y, P2x, P2y, random.randrange(orientations), random.randrange(orientations)])

    @staticmethod
    def _select_block_population(pop):

        randomnet = pop.get_chromosomes()[0].get_net()
        POPULATION_SIZE = len(max(randomnet, key=len))
        random.shuffle(randomnet)
        block_population =  Population(0,randomnet[0:2],pop.get_component(), MinSqAreaGrid, MaxSqAreaGrid)
        i = 0
        while i < GRID_SEL_SIZE:
            block_population.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
            i += 1
            block_population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)

        return block_population

