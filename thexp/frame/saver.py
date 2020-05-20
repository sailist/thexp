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
import json
import os
import re

from ..utils.lazy import torch

from ..utils.generel_util import listdir_by_time
from collections import namedtuple


ckpt_tuple = namedtuple("Checkpoint",["checkpoint",'info'])

# TOOD save 有多种，load只需要一个就可以了，就是直接通过路径load

class Saver:
    _ckpt_fn_templete = "{}{:06d}.ckpt"
    _model_fn_templete = "model.{}{:06d}.pth"
    re_fn = re.compile("^[0-9]{7}\.ckpt$")
    re_keep_fn = re.compile("^keep.[0-9]{7}\.ckpt$")

    def __init__(self, ckpt_dir, max_to_keep=3):
        self.info = {}
        self.ckpt_dir = ckpt_dir
        self.max_to_keep = max_to_keep
        os.makedirs(ckpt_dir, exist_ok=True)

    def _build_checkpoint_name(self, epoch=0, replacement: bool=False, lasting=False):
        i = 0

        def build_fn():
            fn = Saver._ckpt_fn_templete.format(i, epoch)
            if lasting:
                fn = "keep.{}".format(fn)
            return fn

        absfn = os.path.join(self.ckpt_dir, build_fn())
        while replacement == False and os.path.exists(absfn):
            i += 1
            if i >= 9:
                if lasting:
                    kfns = self.find_keypoints()
                else:
                    kfns = self.find_checkpoints()
                return kfns[-1]

            absfn = os.path.join(self.ckpt_dir, build_fn())
        return absfn

    def _build_model_name(self, epoch=0):
        fn = os.path.join(self.ckpt_dir, Saver._model_fn_templete.format(0, epoch))
        return fn

    def _check_max_checkpoint(self):
        fs = self.find_checkpoints()
        while len(fs) > self.max_to_keep:
            self.check_remove(fs[-1])
            fs.pop()

    def find_checkpoints(self):
        """
        find all checkpoint saved in the save dir.
        :return:
        """
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        fs = [os.path.join(self.ckpt_dir, i) for i in fs if re.search(Saver.re_fn, i) is not None]
        return fs

    def find_models(self):
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        fs = [i for i in fs if i.endswith(".pth")]
        return fs

    def find_keypoints(self):
        """
        find all "keeped" checkpoint saved in the save dir.
        :return:
        """
        fs = listdir_by_time(self.ckpt_dir)  # 按创建时间排序
        fs = [os.path.join(self.ckpt_dir, i) for i in fs if re.search(Saver.re_keep_fn, i) is not None]
        return fs

    def save_model(self, epoch:int, state_dict, extra_info: dict = None):
        """
        命名格式为 model.{epoch}.pth , {epoch} 将padding到7位，默认为0
        重复保存会覆盖
        :param epoch:
        :param state_dict:
        :param extra_info:
        :return:
        """
        fn = self._build_model_name(epoch)
        json_fn = "{}.json".format(fn)
        torch().save(state_dict, fn)

        if extra_info is None:
            extra_info = dict()
        extra_info["fn"] = fn

        with open(json_fn, "w", encoding="utf-8") as w:
            json.dump(extra_info, w, indent=2)
        return fn

    def save_checkpoint(self, epoch, state_dict, extra_info: dict = None, replacement: bool = False, lasting=False):
        """
        命名格式为 {epoch}.ckpt ，如果replacement=True，则命名格式为 keep.{epoch}.ckpt
        :param epoch:
        :param state_dict:
        :param extra_info: 额外信息，会以json格式保存在同模型名+ '.json'下
        :param replacement: 如果命名重复是否删除
        :param lasting: 是否受 max_to_keep 影响删除，默认为False
        :return:
        """
        fn = self._build_checkpoint_name(epoch, replacement, lasting)

        json_fn = "{}.json".format(fn)
        torch().save(state_dict, fn)
        if extra_info is None:
            extra_info = dict()
        extra_info["fn"] = fn

        with open(json_fn, "w", encoding="utf-8") as w:
            json.dump(extra_info, w, indent=2)

        if not lasting:
            self._check_max_checkpoint()

        return fn

    def save_keypoint(self, val, state_dict, extra_info: dict = None, replacement: bool = False):
        return self.save_checkpoint(val, state_dict, extra_info, replacement, True)

    def _check_isint_or_str(self, val):
        assert type(val) in {int, str}
        if isinstance(val, int):
            return True
        return False

    def load_latest_checkpoint(self,dir=None):
        if dir is None:
            dir = self.ckpt_dir

        fs = listdir_by_time(dir)
        fs = [os.path.join(dir,i) for i in fs if i.endswith(".ckpt")]
        if len(fs) == 0:
            return ckpt_tuple(None,None)
        return self.load_state_dict(fs[-1])

    def load_state_dict(self, fn):
        """
        :param fn: rel path or abs path
        :return:  None or torch().load() result
        """
        path = self._guess_abs_path(fn)

        if os.path.exists(path):
            ckpt = torch().load(fn)
        else:
            ckpt = None
        info_path = "{}.json".format(path)
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as r:
                info = json.load(r)
        else:
            info = None

        return ckpt_tuple(ckpt, info)

    def _guess_abs_path(self,fn):
        if os.path.basename(fn) == fn:
            path = os.path.join(self.ckpt_dir, fn)
        else:
            path = fn
        return path

    def check_remove(self, fn, with_json=True):
        if os.path.exists(fn):
            os.remove(fn)
        if with_json:
            jfn = "{}.json".format(fn)
            if os.path.exists(jfn):
                os.remove(jfn)

    def clear_models(self):
        fns = filter(lambda x:x.endswith(".pth"),
                     listdir_by_time(self.ckpt_dir))
        for i in fns:
            self.check_remove(os.path.join(self.ckpt_dir,i))

    def clear_checkpoints(self):
        for i in self.find_checkpoints():
            self.check_remove(i)

    def clear_keypoints(self):
        fns = filter(lambda x: x.endswith(".ckpt") and x.startswith("keep."),
                     listdir_by_time(self.ckpt_dir))
        for i in fns:
            self.check_remove(os.path.join(self.ckpt_dir,i))

    def summary(self, detail=False):
        print("checkpoints:")
        print(" || ".join(self.find_checkpoints()))
        print("keppoints:")
        print(" || ".join(self.find_keypoints()))
        print("models:")
        print(" || ".join(self.find_models()))

