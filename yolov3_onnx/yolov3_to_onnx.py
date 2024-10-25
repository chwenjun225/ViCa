from __future__ import print_function 
from collections import OrderedDict
import sys 
import os 

import onnx 
from onnx import helper
from onnx import TensorProto 
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
from downloader import getFilePath 

class DarkNetParser(object):
    """Definition of a parser for DarkNet-based YOLOv3-608 (only tested for this topology)."""
    def __init__(self, supported_layers):
        """Initializes a DarknetParser object.
        
        Keyword argument:
            supported_layers -- a string list of supported layers in DarkNet naming convention, 
            parameters are only added to the class dictionary if a parsed layer is included.
        """
        # A list of YOLOv3 layers containing dictionaries with all layer 
        # parameters:
        self.layer_configs = OrderedDict()
        self.supported_layers = supported_layers
        self.layer_counter = 0 

    def parse_cfg_file(self, cfg_file_path):
        """Takes the yolov3.cfg file and parses it layer by layer, appending each layer's 
        parameters as a dictionary to layer_configs.
        
        Keyword argument:
            cfg_file_path -- path to the yolov3.cfg file as string
        """
        with open(cfg_file_path) as cfg_file:
            remainder = cfg_file.read()
            while remainder is not None:
                layer_dict, layer_name, remainder = self._next_layer(remainder)
                if layer_dict is not None:
                    self.layer_configs[layer_name] = layer_dict
        return self.layer_configs 

    def _next_layer(self, remainder):
        """Takes in a string and segments it by looking for DarkNet delimiters. 
        Returns the layer parameters and the remaining string after the last delimiter.
        Example for the first Conv layer in yolo.cfg ...
        
        [convolutional]
        batch_normalize=1
        filters=32
        size=3 
        stride=1
        pad=1
        activation=leaky

        ... becomes the following layer_dict return value:
        {
            'activation': 'leaky', 
            'stride': 1, 
            'pad': 1, 
            'filters': 32, 
            'batch_normalize': 1, 
            'type': 'convolutional', 
            'size': 3
        }. 

        '001_convolutional' is returned as layer_name, and all lines that follow in yolo.cfg
        are returned as the next remainder.

        Keyword argument:
            remainder -- a string with all raw text after the previously parsed layer
        """
        remainder = remainder.split("[", 1)
        if len(remainder) == 2:
            remainder = remainder[1]
        else:
            return None, None, None 
        remainder = remainder.split("]", 1)
        if len(remainder) == 2:
            layer_type, remainder = remainder 
        else: 
            return None, None, None 
        if remainder.replace(" ", "")[0] == "#":
            remainder = remainder.split("\n", 1)[1]

        layer_param_block, remainder = remainder.split("\n\n", 1)
        layer_param_lines = layer_param_block.split("\n", 1)[1]
        layer_name = str(self.layer_counter).zfill(3) + "_" + layer_type 
        layer_dict = dict(type=layer_type)
        if layer_type in self.supported_layers:
            for param_line in layer_param_lines:
                if param_line[0] == "#":
                    continue 
                param_type, param_value = self._parser_params(param_line)
                layer_dict[param_type] = param_value 
        self.layer_counter += 1 
        return layer_dict, layer_name, remainder 

    def _parser_params(self, param_line):
        """Identifies the parameters contained in one of the cfg file and returns them 
        in the required format for each parameter type, e.g. as a list, and init of float.
        
        Keyword argument:
            param_line -- one parsed line within a layer block 
        """
        param_line = param_line.replace(" ", "")
        param_type, param_value_raw = param_line.split("=")
        param_value = None 
        if param_type == "layers":
            layer_indexes = list()
            for index in param_value_raw.split(","):
                layer_indexes.append(int(index))
            param_value = layer_indexes 
        elif isinstance(param_value_raw, str) and not param_value_raw.isalpha():
            condition_param_value_positive = param_value_raw.isdigit()
            condition_param_value_negative = (
                param_value_raw[0] == "-" and param_value_raw[1:].isdigit()
            )
            if condition_param_value_positive or condition_param_value_negative:
                param_value = int(param_value_raw)
            else:
                param_value = float(param_value_raw)
        else:
            param_value = str(param_value_raw)
        return param_type, param_value

class MajorNodeSpecs(object):
    """Helper class used to store the name of ONNX output names, corresponding
    to the output of a DarkNet layer and its output channels. Some DarkNet layers
    are not created and there is no corresponding ONNX node, but we still need to 
    track them in order to setup skip connection.
    """
    def __init__(self, name, channels):
        """Initialize a MajorNodeSpecs object. 
        
        Keyword arguments:
            name -- name of the ONNX node 
            channels -- number of output channels of this node 
        """        
        self.name = name 
        self.channels = channels 
        self.created_onnx_node = False 
        if name is not None and isinstance(channels, int) and channels > 0:
            self.created_onnx_node = True 

class ConvParams(object):
    """Helper class to store the hyper parameters of a Conv layer, 
    including its prefix name in the ONNX graph and the expected dimensions
    of weights for convoluton, bias, and batch normalization. 
    
    Additionally acts as a wrapper for generating safe names for all weights, 
    checking on feasible combinations.
    """
    def __init__(self, node_name, batch_normalize, conv_weight_dims):
        """Constructors based on the base node name (e.g. 101_convolutional), the batch
        normalization setting, and the convolutional weights shape.
        
        Keyword arguments:
            node_name -- base name of this YOLO convolutional layer 
            batch_normalize -- bool value if batch normalization is used 
            conv_weight_dims -- the dimensions of this layer's convolutional weights
        """
        self.node_name = node_name 
        self.batch_normalize = batch_normalize 
        assert len(conv_weight_dims) == 4
        self.conv_weight_dims = conv_weight_dims 

    def generate_param_name(self, param_category, suffix):
        """Generates a name based on two string inputs, 
        and checks if the combination is valid."""
        assert suffix 
        assert param_category in ["bn", "conv"]
        assert suffix in ["scale", "mean", "var", "weights", "bias"]
        if param_category == "bn": 
            assert self.batch_normalize 
            assert suffix in ["scale", "bias", "mean", "var"]
        elif param_category == "conv":
            assert suffix in ["weights", "bias"]
            if suffix == "bias":
                assert not self.batch_normalize 
        param_name = self.node_name + "_" + param_category + "_" + suffix 
        return param_name 

class ResizeParams(object):
    # Helper class to store the scale parameter for an Resize node. 

    def __init__(self, node_name, value):
        """Constructor based on the base node name (e.g. 86_Resize), 
        and the value of the scale input tensor.
        
        Keyword arguments:
            node_name -- base name of this YOLO Resize layer
            value -- the value of the scale input to the Resize layer as numpy array 
        """
        self.node_name = node_name 
        self.value = value 

    def generate_param_name(self):
        """Generates the scale parameter name for the Resize node."""
        param_name = self.node_name + "_" + "scale"
        return param_name 

    def generate_roi_name(self):
        """Generates the roi input name for the Resize node."""
        param_name = self.node_name + "_" + "roi"
        return param_name 

class WeightLoader(object):
    """Helper class used for loading the serialized weights of a binary file stream 
    and returning the initializers and the input tensors required for populating the 
    ONNX graph with weights."""
    def __init__(self, weights_file_path):
        """Initialized with a path to the YOLOv3 .weith file. 
        
        Keyword arguments:
            weights_file_path -- path to the weights file. 
        """
        self.weights_file = self._open_weights_file(weights_file_path)
    
    def load_resize_scales(self, resize_params):
        """Returns the initializers with the value of the scale input tensor given resize_params.
        
        Keyword arguments:
            resize_params -- a ResizeParams object
        """
        initializers = list()
        inputs = list()
        name = resize_params.generate_param_name()
        shape = resize_params.value.shape 
        data = resize_params.value 
        scale_init = helper.make_tensor(name, TensorProto.FLOAT, shape, data)
        scale_input = helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        initializers.append(scale_init)
        inputs.append(scale_input) 

        # In opset 11 and additional input named roi is required. Create a dummy to tensor
        # to statisfy this. It is a 1D tensor of size of the rank of the input (4)
        rank = 4 
        roi_name = resize_params.generate_roi_name()
        roi_input = helper.make_tensor_value_info(roi_name,TensorProto.FLOAT, [rank])
        roi_init = helper.make_tensor(roi_name, TensorProto.FLOAT, [rank], [0, 0, 0, 0])
        initializers.append(roi_init)
        inputs.append(roi_init)

        return initializers, inputs 

    def load_conv_weights(self, conv_params):
        """Returns the initializers with weights from the weights file and 
        the input tensors of a convolutional layer for all corresponding ONNX nodes.
        
        Keyword arguments:
            conv_params -- a ConvParams object
        """
        initializers = list()
        inputs = list()
        if conv_params.batch_normalize:
            bias_init, bias_input = self._create_param_tensors(
                conv_params, 
                "bn", "bias"
            )
            bn_scale_init, bn_scale_input = self._create_param_tensors(
                conv_params, "bn", "scale"
            )
            bn_mean_init, bn_mean_input = self._create_param_tensors(
                conv_params, "bn", "mean"
            )
            bn_var_init, bn_var_input = self._create_param_tensors(
                conv_params, "bn", "var"
            )
            initializers.extend([bn_scale_init, bias_init, bn_mean_init, bn_var_init])
            inputs.extend([bn_scale_input, bias_input, bn_mean_input, bn_var_input])
        else:
            bias_init, bias_input = self._create_param_tensors(
                conv_params, "conv", "bias"
            )
            initializers.append(bias_init)
            inputs.append(bias_input)
        conv_init, conv_input = self._create_param_tensors(
            conv_params, "conv", "weights"
        )
        initializers.append(conv_init)
        inputs.append(conv_input)
        return initializers, inputs 

    def _open_weights_file(self, weights_file_path):
        """Open a YOLOv3 DarkNet file stream and skips the header. 
        
        Keyword arguments"
            weights_file_path -- path to the weights file. 
        """
        weights_file = open(weights_file_path, "rb")
        length_header = 5 
        np.ndarray(
            shape=(length_header, ), 
            dtype="int32", 
            buffer=weights_file.read(length_header * 4), 
        )
        return weights_file 

    def _create_param_tensors(self, conv_params, param_category, suffix):
        """Creates the initializers with weights from the weights file together
        with the input tensors.

        Keyword arguments:
            conv_params -- a ConvParams object 
            param_category -- the category of parameters to be created ('bn' or 'conv')
            suffix -- a string determining the sub-type of above param_category (e.g., 
                'weights' or 'bias')
        """
        param_name, param_data, param_data_shape = self._load_one_param_type(
            conv_params, param_category, suffix
        )
        initializer_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, param_data_shape, param_data, 
        )
        input_tensor = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, param_data_shape, 
        )
        return initializer_tensor, input_tensor

    def _load_one_param_type(self, conv_params, param_category, suffix):
        """Deserializes the weights from a file stream in the DarkNet order. 

        Keyword arguments:
            conv_param -- a ConvParams object 
            param_category -- the category of parameters to be created ('bn' or 'conv')
            suffix -- a string determining the sub-type of above param_category (e.g., 
                'weights' or 'bias')
        """
        param_name = conv_params.generate_param_name(param_category, suffix)
        channels_out, channels_in, filter_h, filter_w = conv_params.conv_weight_dims 
        if param_category == "bn":
            param_shape = [channels_out]
        elif param_category == "conv": 
            if suffix == "weights": 
                param_shape = [channels_out, channels_in, filter_h, filter_w]
            elif suffix == "bias":
                param_shape = [channels_out]
        param_size = np.product(np.array(param_shape))
        param_data = np.ndarray(
            shape=param_shape, 
            dtype="float32", 
            buffer=self.weights_file.read(param_size * 4), 
        )
        param_data = param_data.flatten().astype(float)
        return param_name, param_data, param_shape 

class GraphBuilderONNX(object):
    """Class for creating an ONNX graph from a previously generated list of layer dictionaries."""
    def __init__(self, output_tensors):
        """Initialize with all DarkNet default parameters used creating YOLOv3, 
        and specify the output tensors as an OrderedDict for their output dimensions
        with their names as keys. 
        
        Keyword arguments:
            output_tensors -- the output tensors as an OrderedDict containing the keys
                output dimensions
        """
        self.output_tensors = output_tensors 
        self._nodes = list()
        self.graph_def = None 
        self.input_tensor = None 
        self.epsilon_bn = 1e-5 
        self.momentum_bn = 0.99 
        self.alpha_lrelu = 0.1 
        self.param_dict = OrderedDict()
        self.major_node_specs = list()
        self.batch_size = 1 

    def build_onnx_graph(self, layer_configs, weights_file_path, verbose=True):
        """Iterate over all layer configs (parsed from the DarkNet representation of 
        YOLOv3-608), create an ONNX graph, populate it with weights from the weights 
        file and return the graph definition.
        Keyword arguments:
            layer_configs -- an OrderedDict object with all parsed layers's configurations
            weights_file_path -- location of the weights file 
            verbose -- toggles if the graph is printed after creation (default: True)
        """
        for layer_name in layer_configs.keys():
            layer_dict = layer_configs[layer_name]
            major_node_specs = self._make_onnx_node(layer_name, layer_dict)
            if major_node_specs.name is not None:
                self.major_node_specs.append(major_node_specs)
        outputs = list()
        for tensor_name in self.output_tensors.keys():
            output_dims = [self.batch_size] + self.output_tensors[tensor_name]
            output_tensor = helper.make_tensor_value_info(
                tensor_name, TensorProto.FLOAT, output_dims, 
            )
            outputs.append(output_tensor)
        inputs = [self.input_tensor]
        weight_loader = WeightLoader(weights_file_path)
        initializer = list()
        # If a layer has parameters, add them to the initializers and input lists
        for layer_name in self.param_dict.keys():
            _, layer_type = layer_name.split("_", 1)
            params = self.param_dict[layer_name]
            if layer_type == "convolutional":
                initializer_layer, inputs_layer = weight_loader.load_conv_weights(
                    params
                )
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)
            elif layer_type == "upsample":
                initializer_layer, inputs_layer = weight_loader.load_resize_scales(
                    params
                )
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)
        del weight_loader 
        self.graph_def = helper.make_graph(
            nodes=self._nodes, 
            name="YOLOv3-608", 
            inputs=inputs, 
            outputs=outputs, 
            initializer=initializer
        )
        if verbose:
            print(helper.printable_graph(self.graph_def))
        model_def = helper.make_model(
            self.graph_def, 
            producer_name="Vision Clarify"
        )
        return model_def 

    def _make_onnx_node(self, layer_name, layer_dict):
        pass 

    def _make_input_tensor(self, layer_name, layer_dict):
        pass 

    def _get_previous_node_specs(self, target_index=-1):
        pass 

    def _make_conv_node(self, layer_name, layer_dict):
        pass 

    def _make_shortcut_node(self, layer_name, layer_dict):
        pass 

    def _make_route_node(self, layer_name, layer_dict):
        pass 

    def _make_resize_node(self, layer_name, layer_dict):
        pass 

def main():
    pass 

if __name__ == '__main__':
    main()
