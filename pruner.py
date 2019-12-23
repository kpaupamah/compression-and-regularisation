"""
Code adapted from https://github.com/synxlin/nn-compression/
"""

import torch
import math
import numpy as np
import re


def prune_elementwise(param, sparsity, fn_importance=lambda x: x.abs()):
    """
    element-wise vanilla pruning
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
    :param fn_importance: function, inputs 'param' and returns the importance of
                                    each position in 'param',
                                    default=lambda x: x.abs()
    :return:
        torch.(cuda.)ByteTensor, mask for zeros
    """

    sparsity = min(max(0.0, sparsity), 1.0)

    if sparsity == 1.0:
        return torch.zeros_like(param).bool()

    num_el = param.numel()
    importance = fn_importance(param)
    num_pruned = int(math.ceil(num_el * sparsity))
    num_stayed = num_el - num_pruned

    if sparsity <= 0.5:
        _, topk_indices = torch.topk(importance.view(num_el), k=num_pruned, dim=0, largest=False, sorted=False)
        mask = torch.zeros_like(param).bool()
        param.view(num_el).index_fill_(0, topk_indices, 0)
        mask.view(num_el).index_fill_(0, topk_indices, 1)
    else:
        thr = torch.min(torch.topk(importance.view(num_el), k=num_stayed, dim=0, largest=True, sorted=False)[0])
        mask = torch.lt(importance, thr)
        param.masked_fill_(mask, 0)
    return mask


def prune_filterwise(sparsity, param, fn_importance=lambda x: x.norm(1, -1)):
    """
    filter-wise vanilla pruning, the importance determined by L1 norm
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
    :param fn_importance: function, inputs 'param' as size (param.size(0), -1) and
                                returns the importance of each filter in 'param',
                                default=lambda x: x.norm(1, -1)
    :return:
        torch.(cuda.)ByteTensor, mask for zeros
    """
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).bool()
    num_filters = param.size(0)
    param_k = param.view(num_filters, -1)
    param_importance = fn_importance(param_k)
    num_pruned = int(math.ceil(num_filters * sparsity))
    _, topk_indices = torch.topk(param_importance, k=num_pruned,
                                 dim=0, largest=False, sorted=False)
    mask = torch.zeros_like(param).bool()
    mask_k = mask.view(num_filters, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask


class Pruner(object):

    def __init__(self, rule=None):
        """
        Pruner Class for Vanilla Pruning Method
        :param rule: str, path to the rule file, each line formats
                          'param_name granularity sparsity_stage_0, sparstiy_stage_1, ...'
                     list of tuple, [(param_name(str), granularity(str),
                                      sparsity(float) or [sparsity_stage_0(float), sparstiy_stage_1,],
                                      fn_importance(optional, str or function))]
                     'granularity': str, choose from ['element', 'filter']
                     'fn_importance': str, choose from ['abs', 'l2norm']
        e.g:
        rule = [
            ['0.weight', [0.3, 0.5], 'abs'],
            ['1.weight', [0.4, 0.6], 'default'],
            ['2.weight', [0.5, 0.7], 'l2norm']
        ]
        """

        self.rule = rule
        self.masks = dict()

        print("=" * 89)
        if self.rule:
            print("Initializing Pruner with rules:")
            print(self.rule)
        else:
            print("Initializing Pruner WITHOUT rules")
        print("=" * 89)

    def prune_param(self, param, param_name, stage=0, verbose=False):
        """
        prune parameter
        :param param: torch.(cuda.)tensor
        :param param_name: str, name of param
        :param stage: int, the pruning stage, default=0
        :param verbose: bool, whether to print the pruning details
        :return:
            torch.(cuda.)ByteTensor, mask for zeros
        """
        rule_id = -1
        for idx, r in enumerate(self.rule):
            m = re.match(r[0], param_name)
            if m is not None and len(param_name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            sparsity = self.rule[rule_id][1][stage]
            fn_importance = self.rule[rule_id][2]
            if verbose:
                print("{param_name:^30} | {stage:5d} | {spars:.3f}".
                      format(param_name=param_name, stage=stage, spars=sparsity))
            if fn_importance is None or fn_importance == 'default':
                mask = prune_elementwise(param=param, sparsity=sparsity)
            elif fn_importance == 'abs':
                mask = prune_elementwise(param=param, sparsity=sparsity, fn_importance=lambda x: x.abs())
            elif fn_importance == 'l2norm':
                mask = prune_filterwise(param=param, sparsity=sparsity, fn_importance=lambda x: x.norm(2, -1))
            else:
                mask = prune_elementwise(param=param, sparsity=sparsity, fn_importance=fn_importance)
            return mask
        else:
            if verbose:
                print("{param_name:^30} | skipping".format(param_name=param_name))
            return None

    def prune(self, model, stage=0, update_masks=False, verbose=False):
        """
        prune models
        :param model: torch.nn.Module
        :param stage: int, the pruning stage, default=0
        :param update_masks: bool, whether update masks
        :param verbose: bool, whether to print the pruning details
        :return:
            void
        """
        update_masks = True if update_masks or len(self.masks) == 0 else False
        if verbose:
            print("=" * 89)
            print("Pruning Models")
            if len(self.masks) == 0:
                print("Initializing Masks")
            elif update_masks:
                print("Updating Masks")
            print("=" * 89)
            print("{name:^30} | stage | sparsity".format(name='param_name'))
        for param_name, param in model.named_parameters():
            if 'AuxLogits' not in param_name:
                # deal with googlenet
                if param.dim() > 1:
                    if update_masks:
                        mask = self.prune_param(param=param.data, param_name=param_name,
                                                stage=stage, verbose=verbose)
                        param.data.masked_fill_(mask, 0)
                        if mask is not None:
                            self.masks[param_name] = mask
                    else:
                        if param_name in self.masks:
                            mask = self.masks[param_name]
                            param.data.masked_fill_(mask, 0)
        if verbose:
            print("=" * 89)