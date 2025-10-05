from unittest import TestCase
from src.qlora_finetune_base.models.model_loader import load_model
from src.qlora_finetune_base.models.lora_peft import LoRAAdapter

class TestModelLoading(TestCase):
    def test_load_model(self):
        model_name = "tiny-llama"
        model = load_model(model_name)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, model_name)

class TestLoRAFunctionality(TestCase):
    def test_lora_adapter(self):
        base_model = load_model("tiny-llama")
        lora_adapter = LoRAAdapter(base_model)
        self.assertIsNotNone(lora_adapter)
        self.assertEqual(lora_adapter.base_model.name, base_model.name)

    def test_lora_integration(self):
        base_model = load_model("tiny-llama")
        lora_adapter = LoRAAdapter(base_model)
        integrated_model = lora_adapter.integrate()
        self.assertIsNotNone(integrated_model)
        self.assertNotEqual(integrated_model, base_model)