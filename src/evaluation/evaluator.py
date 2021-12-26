from abstractions import EvaluatorBase
from aimedeval.segmentation import IoUScore, SoftIoUScore, DiceScore, SoftDice, TPR, TNR, Precision


class Evaluator(EvaluatorBase):
    def get_eval_funcs(self):

        threshold = 0.5

        tpr = TPR(threshold=threshold)
        tnr = TNR(threshold=threshold)
        dice = DiceScore(threshold=threshold)
        soft_dice = SoftDice()
        iou = IoUScore(threshold=threshold)
        soft_iou = SoftIoUScore()

        funcs = [tpr, tnr, dice, soft_dice, iou, soft_iou]

        return funcs
