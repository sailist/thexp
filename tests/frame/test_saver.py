"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""

from thexp.frame.saver import Saver

saver = Saver("./sav")

def test_max_to_keep():
    for i in range(5):
        saver.save_checkpoint(i,dict(a=i),dict(b=i))
    assert len(saver.find_checkpoints()) == saver.max_to_keep
    saver.clear_checkpoints()
    assert len(saver.find_checkpoints()) == 0

def test_keypoint():
    saver.clear_keypoints()
    for i in range(5):
        saver.save_keypoint(i,dict(a=i),dict(b=i))
    for i in range(5):
        assert saver.load_keypoint(i)["a"] == i
        assert saver.load_keypoint_info(i)["b"] == i

    assert len(saver.find_keypoints()) == 5
    saver.clear_keypoints()
    assert len(saver.find_keypoints()) == 0
