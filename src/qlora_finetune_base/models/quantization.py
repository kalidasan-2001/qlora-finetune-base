def quantize_model_weights(model, num_bits=4):
    """
    Quantizes the model weights to the specified number of bits.
    
    Args:
        model: The model whose weights are to be quantized.
        num_bits: The number of bits for quantization (default is 4).
        
    Returns:
        A new model with quantized weights.
    """
    # Ensure the number of bits is valid
    if num_bits not in [2, 4, 8]:
        raise ValueError("num_bits must be one of [2, 4, 8]")
    
    quantized_model = model  # Placeholder for the quantized model
    
    # Implement the quantization logic here
    # This is a simplified example and should be replaced with actual quantization logic
    for param in model.parameters():
        param.data = (param.data * (2 ** num_bits)).round() / (2 ** num_bits)
    
    return quantized_model


def dequantize_model_weights(quantized_model):
    """
    Dequantizes the model weights back to their original precision.
    
    Args:
        quantized_model: The model with quantized weights.
        
    Returns:
        A new model with dequantized weights.
    """
    dequantized_model = quantized_model  # Placeholder for the dequantized model
    
    # Implement the dequantization logic here
    # This is a simplified example and should be replaced with actual dequantization logic
    for param in quantized_model.parameters():
        param.data = param.data.float()  # Convert back to float
    
    return dequantized_model