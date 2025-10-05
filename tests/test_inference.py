import pytest
from src.qlora_finetune_base.inference.generate import generate_text

def test_generate_text():
    prompt = "Once upon a time"
    expected_output = "Once upon a time, there was a brave knight."
    
    # Assuming generate_text returns a string
    output = generate_text(prompt)
    
    assert isinstance(output, str)
    assert expected_output in output  # Check if the output contains the expected text

def test_generate_text_with_empty_prompt():
    prompt = ""
    
    output = generate_text(prompt)
    
    assert output == "Prompt cannot be empty."  # Assuming the function handles empty prompts this way

def test_generate_text_with_long_prompt():
    prompt = "A long prompt that exceeds the usual length" * 10  # Create a long prompt
    
    output = generate_text(prompt)
    
    assert isinstance(output, str)  # Check if the output is still a string
    assert len(output) > 0  # Ensure that some output is generated

def test_generate_text_with_special_characters():
    prompt = "What happens when you mix science and magic?"
    
    output = generate_text(prompt)
    
    assert isinstance(output, str)
    assert "science" in output  # Check if the output contains the word "science"