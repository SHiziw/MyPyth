import nuralNetworks as nwk
iaaa = 3
ia = 3
iaa = 3
ai = 0.3
n = nwk.neuralNetwork(ia, iaa, iaaa, ai)
n.query([1.0, 0.5, -1.5])
