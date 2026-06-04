import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(path):
    return (ROOT / path).read_text(encoding="utf-8")


class ProjectHygieneTests(unittest.TestCase):
    def test_readme_references_existing_entrypoint_scripts(self):
        readme = read_text("ReadMe.md")

        self.assertNotIn("train_trainsitionNet.py", readme)
        self.assertNotIn("benchmarks.py", readme)
        self.assertIn("train_transitionNet.py", readme)
        self.assertIn("benchmark.py", readme)
        self.assertIn("scripts/check_setup.py", readme)
        self.assertIn("scripts/smoke_verify.py", readme)


    def test_style_loader_load_part_uses_configured_root_dir(self):
        source = read_text("src/Datasets/Style100Processor.py")
        tree = ast.parse(source)

        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_part_to_binary"
        )
        method_source = ast.get_source_segment(source, method)

        self.assertNotIn('path = "./"', method_source)
        self.assertIn("self.root_dir", method_source)
        self.assertIn("os.path.join", method_source)


    def test_train_stylevae_preserves_resume_checkpoint_argument(self):
        for script in ["train_deephase.py", "train_styleVAE.py", "train_transitionNet.py"]:
            source = read_text(script)
            tree = ast.parse(source)
            function = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name in {"training_style100", "training_style100_phase"}
            )
            function_source = ast.get_source_segment(source, function)

            self.assertNotIn("resume_from_checkpoint = None", function_source)
            self.assertIn("fit_with_checkpoint(trainer, model, data_module, resume_from_checkpoint)", source)

    def test_train_scripts_use_lightning_compat_wrapper(self):
        deprecated_trainer_args = [
            "auto_select_gpus",
            "flush_logs_every_n_steps",
            "weights_summary",
            "resume_from_checkpoint=",
        ]
        for script in ["train_deephase.py", "train_styleVAE.py", "train_transitionNet.py"]:
            source = read_text(script)

            self.assertIn("create_trainer", source)
            self.assertIn("fit_with_checkpoint", source)
            for deprecated_arg in deprecated_trainer_args:
                self.assertNotIn(deprecated_arg, source)


    def test_train_entrypoints_choose_cpu_when_cuda_is_unavailable(self):
        for script in ["train_deephase.py", "train_styleVAE.py", "train_transitionNet.py"]:
            source = read_text(script)
            tree = ast.parse(source)
            function = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "select_gpu_par"
            )
            function_source = ast.get_source_segment(source, function)

            self.assertIn("torch.cuda.is_available()", function_source)
            self.assertIn('"accelerator": "cpu"', function_source)

    def test_train_entrypoints_do_not_hardcode_cuda_in_test_mode(self):
        for script in ["train_deephase.py", "train_styleVAE.py", "train_transitionNet.py"]:
            source = read_text(script)

            self.assertNotIn(".cuda(", source)
            self.assertIn("get_default_device", source)


    def test_running_long_seq_accepts_runtime_paths_from_cli(self):
        source = read_text("Running_LongSeq.py")
        tree = ast.parse(source)

        load_model = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_model"
        )
        load_model_args = [arg.arg for arg in load_model.args.args]

        self.assertIn("model_path", load_model_args)
        self.assertIn("ArgumentParser", source)
        self.assertIn("--model_path", source)
        self.assertIn("--source_bvh", source)

    def test_running_long_seq_uses_default_device_instead_of_hardcoded_cuda(self):
        source = read_text("Running_LongSeq.py")

        self.assertNotIn(".cuda(", source)
        self.assertIn("get_default_device", source)

    def test_checkpoint_regex_digit_escape_is_raw(self):
        for script in ["train_styleVAE.py", "train_transitionNet.py", "src/utils/locate_model.py"]:
            source = read_text(script)

            self.assertNotIn('-step=\\d+.ckpt"', source)
            self.assertIn(r"-step=\d+\.ckpt", source)

    def test_dataset_pipeline_uses_default_device_instead_of_hardcoded_cuda(self):
        for script in [
            "add_phase_to_dataset.py",
            "src/Datasets/DeepPhaseDataModule.py",
            "src/Datasets/Style100Processor.py",
            "src/Datasets/StyleVAE_DataModule.py",
        ]:
            source = read_text(script)

            self.assertNotIn(".cuda(", source)
            self.assertIn("get_default_device", source)

    def test_benchmark_and_model_helpers_use_default_device(self):
        for script in ["benchmark.py", "src/Net/TransitionPhaseNet.py"]:
            source = read_text(script)

            self.assertNotIn(".cuda(", source)
            self.assertIn("get_default_device", source)

    def test_setup_and_smoke_scripts_exist(self):
        check_setup = read_text("scripts/check_setup.py")
        smoke = read_text("scripts/smoke_verify.py")

        self.assertIn("def main", check_setup)
        self.assertIn("MotionData/100STYLE", check_setup)
        self.assertIn("torch", check_setup)
        self.assertIn("pytorch_lightning", check_setup)
        self.assertIn("pytorch3d", check_setup)
        self.assertIn("def main", smoke)
        self.assertIn("unittest discover", smoke)
        self.assertIn("py_compile", smoke)

    def test_legacy_style_benchmark_has_cli_and_default_device(self):
        source = read_text("benchmarkStyle100_withStyle.py")

        self.assertIn("ArgumentParser", source)
        self.assertIn("--model_path", source)
        self.assertIn("--model_name", source)
        self.assertIn("get_default_device", source)
        self.assertNotIn("torch.load('./results/", source)


if __name__ == "__main__":
    unittest.main()
