import inspect


def _signature_supports(callable_obj, name):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return True
    return name in signature.parameters


def _filter_supported_kwargs(callable_obj, kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def create_trainer(trainer_cls, checkpoint_path=None, **kwargs):
    resume_in_init = bool(checkpoint_path) and _signature_supports(
        trainer_cls.__init__, "resume_from_checkpoint"
    )
    if resume_in_init:
        kwargs["resume_from_checkpoint"] = checkpoint_path
    trainer = trainer_cls(**_filter_supported_kwargs(trainer_cls.__init__, kwargs))
    trainer._rsmt_resume_from_init = resume_in_init
    return trainer


def fit_with_checkpoint(trainer, model, datamodule=None, checkpoint_path=None):
    fit_kwargs = {"datamodule": datamodule}
    if (
        checkpoint_path
        and not getattr(trainer, "_rsmt_resume_from_init", False)
        and _signature_supports(trainer.fit, "ckpt_path")
    ):
        fit_kwargs["ckpt_path"] = checkpoint_path
    fit_kwargs = _filter_supported_kwargs(trainer.fit, fit_kwargs)
    return trainer.fit(model, **fit_kwargs)
