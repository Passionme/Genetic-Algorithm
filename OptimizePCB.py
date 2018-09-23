import os
import json
import numpy as np
from utils.grid import Grid
from utils.component import Component
from utils.part import Part
from utils.pin import Pin
from GA import Population,GeneticAlgorithm
import tracemalloc

global pinArray

class Input_Output():

    def __init__(self):
        self.Ifilename = "./input.json"
        self.Ofilename = "./output.json"
        self.dir    =   os.path.dirname(__file__)

    def readInput(self):

        def checking_for_unique_keys(pairs):
            result = dict()
            for key, value in pairs:
                if key in result:
                  print("duplicate key ('%s') specified in %s" % (key, self.Ifilename), KeyError)
                result[key] = value
            return result

        try:
            with open(self.Ifilename, 'r') as f:
                json_data = json.load(f, object_pairs_hook=checking_for_unique_keys)
        except (IOError, OSError):
                 print("Couldn't open JSON file: %s" % self.Ifilename, IOError)

        return json_data

    def saveToOutput(self,data):

        try:
            with open(self.Ifilename, 'w') as f:
                json_data = json.dumps(f, data)

        except (IOError, OSError):
            print("Couldn't write to JSON file: %s" % self.Ofilename, IOError)

def deriveComponents():

    grids.setNoOfComponents(len(config["parts"]))

    for each in range(grids.getNoOfComponents()):
        # Define component
        comp = Component()
        comp._name = "Component" + str(each + 1)
        comp._idx = each
        comp._parts_names = config["parts"][each]
        comp._parts_pos = config["parts_name_position"][each]
        comp._parts_size = np.array(config["parts_name_size"][each])
        comp._parts_max_dist = config["parts_max_distance"][each]
        comp._parts_min_dist = config["parts_min_distance"][each]
        comp._coordinate = config["Component_position"][each]
        if len(config["net_points"]) > each:
            comp._netpoints = (config["net_points"][each])
        else:
            comp._netpoints = []
        grids.addComponents(C = comp)

        for idx, eachpart in enumerate(comp.getPartsNames()):
            # Define Parts of component
            part = Part()
            part._name = eachpart
            part._idx = idx
            part._size = config["parts_size"][comp._idx][idx]  # [dx, dy]
            part._NoOfPins = len(config["pins_data"][comp._idx][idx])
            part._pins_data = config["pins_data"][comp._idx][idx]
            part._coordinate = config["parts_size"][comp._idx][idx]
            comp.addPart(part)

        for idx, eachPinData in enumerate(part.getPinsData()):
            # Define pins of Part
            pin = Pin()
            pin._name = "Pin" + str(idx)
            pin._ref = eachPinData
            pin._pinNo = idx
            pin._type = eachPinData[0]
            pin._coordinates = [eachPinData[1], eachPinData[2]]
            if pin.getPinType() == 0:
                pin._dx = eachPinData[3]
                pin._dy = eachPinData[4]
            else:
                pin._diameter = eachPinData[3]
                pin._shape = eachPinData[4]


def convertToMatrix(pinsdata=None, parts_size = None):

    global pinArray
    Sq = []
    # for sq_no, each_sq in enumerate(pinsdata):
    X = {}
    Y = {}
    pinarray = np.empty(shape=(parts_size[0], parts_size[1]))
    pinarray.fill(-1)
    # print("PinNo:   |      ", "PinData:")
    # print("--------------------------")
    for pin_no, pindata in enumerate(pinsdata):
        # print(pin_no, pindata)

        X.setdefault(pindata[1], []).append((pin_no, pindata))
        Y.setdefault(pindata[2], []).append((pin_no, pindata))
    # X = { k:v for k,v in X.items() if len(v)> 2}
    ascend = {}
    for k, v in X.items():
        l = v
        l.sort(key=lambda l: l[1][2])
        ascend.update({k: l})
    X = ascend
    Sq.append({"x": X, "y": Y})
    print("Matrix form of PinData")
    print('-----------------------')
    for ix, d in X.items():
        x = int(ix)
        for data in d:
            pinno = data[0]
            y = int(data[1][2])

            if pinarray[x, y] != -1:
                if pinarray[x-1, y] != -1:
                    x = x+1
                elif  pinarray[x, y-1] != -1:
                    y = y + 1
            pinarray.itemset(x, y, pinno)
            # print(pinno," ", [data[1][0],x,y,data[1][3],data[1][4]])
            y_prev = data[1][2]
    pinArray.append(pinarray)
    # print(pinarray)
    return pinarray


if __name__ == '__main__' :
    tracemalloc.start()

    IO = Input_Output()
    grids = Grid()

    config = IO.readInput()
    deriveComponents()

    for Comp in grids.getComponents():
        parts_min_distance = Comp.getPartsMinDist()
        parts_max_distance = Comp.getPartsMaxDist()
        net_points         =  [Comp.getNetPoints()]
        NoOfSquares = Comp.getNoOfParts()

        parts_size = Comp.getPartsSize()

        pinArray = []

        for eachPart in Comp.getParts():
            pinsdata = eachPart.getPinsData()
            partssize = eachPart.getPartsSize()

            eachPart._Matrix = convertToMatrix(pinsdata, partssize)
            eachPart.printMatrix()
        pinArray_rotated = []
        parts_size_rotated = []
        MinSqAreaGrid = 0

        for Sq in parts_size:
            if len(Sq.shape) == 1:
                MinSqAreaGrid += (Sq.shape[0])
            else:
                MinSqAreaGrid += (Sq.shape[0] + Sq.shape[1])

        for dist in parts_min_distance:
            MinSqAreaGrid += dist

        MaxSqAreaGrid = MinSqAreaGrid
        for dist in parts_max_distance:
            MaxSqAreaGrid += dist
        if len(net_points) > 0:
            POPULATION_SIZE = len(max(net_points, key=len))
            NUM_OF_ELITE_CHROMOSOMES = 1
            GRID_SEL_SIZE = 4
            MUTATION_RATE = 0.01
            orientations = 4  # 0,90,180,270
            for idx, net in enumerate(net_points):
                net_i = net
                i = 0
                gen = 0
                while len(net_i) > 1:
                    print("=====================================")

                    print("net in consideration",net_i)
                    POPULATION_SIZE = orientations * orientations     # orientations sq1 * Orientation of sq2
                    population = Population(POPULATION_SIZE,net_i[i:i+2],grids.getComponents()[idx], MinSqAreaGrid, MaxSqAreaGrid)
                    population.get_chromosomes()
                    # population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=True)
                    # population.get_chromosomes().sort(key=lambda x:x.get_pinDist(), reverse=False)
                    population._print_population(gen,i)

                    # while population.get_chromosomes()[0].get_fitness() < 5:
                    #     population = GeneticAlgorithm.evolve(population)
                    #     population.get_chromosomes().sort(key=lambda x:x.get_fitness(), reverse=True)
                    #     population._print_population( gen,i)
                    #     # population._print_population(population, gen)
                    # #     gen += 1
                    #
                    # net_i = np.delete(net_i, i+1,0)

                    snapshot = (tracemalloc.take_snapshot())
                    top_stats = snapshot.statistics('lineno')

                    for index, stat in enumerate(top_stats[:], 1):
                        total = sum(stat.size for stat in top_stats)
                    print("Total allocated size: %.1f KiB" % (total / 1024))
        else:
            print("Warning! No net points connecting Component '",Comp.getName(), "'.")
