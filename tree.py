class Node:

    def __init__(self, value):
        self.value = value
        self.branches = []
        self.children = []

    def split(self, branch, newNode):
        self.branches.append(branch)
        self.children.append(newNode)

    # Not the best idea to use this if the tree gets too big
    def printTree(self):
        if len(self.branches) == 0:
            return self.value
        else:
            treeRep = {}
            treeRep[self.value] = {}
            for split in range(len(self.branches)):
                treeRep[self.value][self.branches[split]] = self.children[split].printTree()
            return treeRep

# Given a data instance and tree, this will follow the tree to a leaf for classification
def followTree(tree, dataInstance, featureSet, categories):
    treeRoot = tree.value
    if treeRoot in categories:
        return treeRoot
    else:
        attr = treeRoot
        featValue = dataInstance[featureSet.index(attr)]
        branchIndex = tree.branches.index(featValue)
        followChild = tree.children[branchIndex]

        return followTree(followChild, dataInstance, featureSet, categories)

def treeDepth(tree, depth):
    vals = []
    if len(tree.children)==0:
        return depth
    else:
        return max([treeDepth(tree.children[0], depth+1), treeDepth(tree.children[1], depth+1)])

def numNodes(tree, num):
    if len(tree.children) == 0:
        return num
    else:
        return sum([numNodes(tree.children[0], num+1), numNodes(tree.children[1], num+1)])