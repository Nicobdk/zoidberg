import tensorflow as tf

model = tf.keras.models.load_model("notebooks/03_modeling/efficientnet_binary_stage1.keras")

print("--- MODEL LAYERS ---")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({type(layer).__name__})")
    if isinstance(layer, tf.keras.Model):
        print("  This is a sub-model!")
        for j, sub_layer in enumerate(reversed(layer.layers)):
            has_shape = hasattr(sub_layer, 'output_shape')
            shape = sub_layer.output_shape if has_shape else "N/A"
            if j < 5: # Just print the last 5 layers of the sub-model
                print(f"    Sub-layer: {sub_layer.name} ({type(sub_layer).__name__}), shape: {shape}")

    has_shape = hasattr(layer, 'output_shape')
    shape = layer.output_shape if has_shape else "N/A"
    print(f"  Shape: {shape}")

