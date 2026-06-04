import unittest

from src.utils.lightning_compat import create_trainer, fit_with_checkpoint


class TrainerV1:
    def __init__(self, max_epochs=None, resume_from_checkpoint=None):
        self.max_epochs = max_epochs
        self.resume_from_checkpoint = resume_from_checkpoint
        self.fit_calls = []

    def fit(self, model, datamodule=None):
        self.fit_calls.append({"model": model, "datamodule": datamodule})
        return "fit-v1"


class TrainerV2:
    def __init__(self, max_epochs=None):
        self.max_epochs = max_epochs
        self.fit_calls = []

    def fit(self, model, datamodule=None, ckpt_path=None):
        self.fit_calls.append({"model": model, "datamodule": datamodule, "ckpt_path": ckpt_path})
        return "fit-v2"


class LightningCompatTests(unittest.TestCase):
    def test_create_trainer_passes_resume_to_legacy_trainer_init(self):
        trainer = create_trainer(
            TrainerV1,
            checkpoint_path="legacy.ckpt",
            max_epochs=5,
            unsupported_option=True,
        )

        self.assertEqual(5, trainer.max_epochs)
        self.assertEqual("legacy.ckpt", trainer.resume_from_checkpoint)
        self.assertTrue(trainer._rsmt_resume_from_init)

    def test_fit_with_checkpoint_uses_ckpt_path_when_fit_supports_it(self):
        trainer = create_trainer(
            TrainerV2,
            checkpoint_path="modern.ckpt",
            max_epochs=9,
            unsupported_option=True,
        )

        result = fit_with_checkpoint(trainer, "model", "data", "modern.ckpt")

        self.assertEqual("fit-v2", result)
        self.assertEqual(9, trainer.max_epochs)
        self.assertEqual(
            [{"model": "model", "datamodule": "data", "ckpt_path": "modern.ckpt"}],
            trainer.fit_calls,
        )


if __name__ == "__main__":
    unittest.main()
