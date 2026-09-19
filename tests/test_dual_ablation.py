"""Guard matched-time stopping and reconstruction split contamination."""

import runpy
from pathlib import Path

import pytest
import torch

helpers=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_dual_ablation.py'))


def test_compute_control_waits_for_budget_and_stops_only_at_final_budget():
    decide=helpers['budget_decision']
    status={'status':'running','compute_seconds':10,'training_budget_final':False}
    assert decide(9,status)=='train'
    assert decide(10,status)=='wait'
    assert decide(11,status)=='wait'
    status['training_budget_final']=True
    assert decide(9,status)=='train'
    assert decide(10,status)=='stop'
    assert decide(11,status)=='stop'
    status['status']='failed'
    with pytest.raises(RuntimeError):decide(11,status)


def test_reconstruction_split_keeps_aliases_together_and_excludes_ineligible():
    bank=torch.zeros(6,1,32,32,dtype=torch.uint8)
    for i in range(1,5):bank[i,0,0,i]=1
    bank[5]=bank[1]
    left,right=helpers['glyph_partition'](bank,[1,2,3,4,5],519,.25)
    assert len(left)==3 and len(right)==1
    assert 0 not in left+right
    assert all(not torch.equal(bank[a],bank[b]) for a in left for b in right)
    assert (left,right)==helpers['glyph_partition'](bank,[1,2,3,4,5],519,.25)
