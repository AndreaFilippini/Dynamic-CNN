# Dynamic-CNN
This module allows you to perform operations on the structure of convolutional neural networks, specifically:
* delete layers
* insert layers
* replace layers

# Examples
Once you have imported and instazied the class, you can perform operations on your model. 

**Replace all MaxPool layers with AveragePool**
```python
model = dynamicNet.insert_section(model, 1, [AveragePooling2D()], 'replace', 'MaxPooling2D')
```

**Add a 'dense section' after the Flatten made up of a Dense layer, an Activation and Dropout**
```python
new_section = [Dense(50), Activation('relu'), Dropout(0.1)]
model = dynamicNet.insert_section(model, 1, new_section, 'after', 'Flatten')
```

**Add a 'convolutional section' before the last one**
```python
new_section = [Conv2D(1024, (2,2), padding="same"), Conv2D(1024, (2,2), padding="same"), MaxPooling2D()]
last_conv_start = dynamicNet.get_last_section(model, 'Conv2D')
model = dynamicNet.insert_section(model, 1, new_section, 'before', last_conv_start)
```

**Remove a 'convolutional section' that starts from 'block5_conv1' with all associated layers in linked_section**
```python
linked_section = ['Conv2D', 'MaxPooling2D']
model = dynamicNet.remove_section(model, 'block5_conv1', linked_section, True, False)
```

# Limitations
Currently, the module only supports convolutional networks, also some layers may not be supported
