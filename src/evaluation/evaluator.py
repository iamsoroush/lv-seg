from abstractions import EvaluatorBase
from abstractions.evaluation import EvalFuncs

from .eval_funcs import get_tpr, get_tnr, get_dice_coeff, get_soft_dice, get_iou_coef, get_soft_iou


class Evaluator(EvaluatorBase):
    def get_eval_funcs(self) -> EvalFuncs:

        tpr = get_tpr(threshold=0.5)
        tnr = get_tnr(threshold=0.5)
        dice = get_dice_coeff(threshold=0.5)
        soft_dice = get_soft_dice()
        iou = get_iou_coef(threshold=0.5)
        soft_iou = get_soft_iou()

        funcs = {tpr.__name__: tpr,
                 tnr.__name__: tnr,
                 dice.__name__: dice,
                 soft_dice.__name__: soft_dice,
                 iou.__name__: iou,
                 soft_iou.__name__: soft_iou}

        return funcs
