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
        source = read_text("train_styleVAE.py")
        tree = ast.parse(source)
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "training_style100"
        )
        function_source = ast.get_source_segment(source, function)

        self.assertNotIn("resume_from_checkpoint = None", function_source)
        self.assertIn("resume_from_checkpoint=resume_from_checkpoint", source)


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


if __name__ == "__main__":
    unittest.main()
