from models.base import base
from models.l2x import l2x
from models.cl2x import cl2x
from models.invase import invase
from models.diff import diff
from models.lasso import lasso
from models.tree import tree
from models.diff_v2 import diff_v2


models = {
    "Base": base,
    "L2X": l2x,
    "CL2X": cl2x,
    "INVASE": invase,
    "DIFF": diff,
    "LASSO": lasso,
    "TREE": tree,
    "DIFF_V2": diff_v2,
}
