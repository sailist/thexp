"""
_licence_content_
"""

from trainers._trainer_name_ import _trainer_name_Param,_trainer_name_Trainer


if __name__ == '__main__':
    param = _trainer_name_Param()
    param.build_exp_name(["arch"], prefix="_trainer_name_")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    trainer = _trainer_name_Trainer(param)
    # trainer.add_log_path()
    trainer.logger.inline(param)

    ti = Traininfo()
    ti.auto_hook(trainer)

    # drawcall = DrawCallBack(None)
    # drawcall.auto_hook(trainer)

    # ec = ExpCheckpoint(100)
    # ec.auto_hook(trainer)

    # sv = ModelCheckpoint("all_loss")
    # sv.auto_hook(trainer)

    # trainer.load_checkpoint()

    trainer.train()
