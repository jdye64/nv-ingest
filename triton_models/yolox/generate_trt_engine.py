import tensorrt as trt
import torch
import yolox_l_9
from tqdm import tqdm


def retrieve_yolox_model(current_exp, ckpt_file, size=(1024, 1024), verbose=1):
    """
    Retrieves and configures a YOLOX model for inference.

    Args:
        exp_file (str): The path to the experiment file.
        ckpt_file (str): The path to the checkpoint file containing the model weights.
        size (tuple, optional): The input size of the model. Defaults to (1024, 1024).
        verbose (int, optional): Verbosity level. If 1, it prints the loading message. Defaults to 1.

    Returns:
        nn.Module: The configured YOLOX model for inference.
    """

    exp = current_exp.Exp()
    exp.test_conf = 0.0
    exp.test_size = size
    exp.nmsthre = 0.75

    model_roi_ = exp.get_model()

    if verbose:
        print(" -> Loading weights from", ckpt_file)

    ckpt = torch.load(ckpt_file, map_location="cpu")
    model_roi_.load_state_dict(ckpt["model"], strict=True)

    model_roi_.max_det = 100
    model_roi_.nmsthre = 0.75
    model_roi_.test_conf = 0.1
    model_roi_.test_size = exp.test_size
    model_roi_.num_classes = 1
    model_roi_.stride = 64
    model_roi_.amp = False  # FP16

    return model_roi_.eval().cuda()


def export2onnx(current_exp, ckpt_file, onnx_filename="model.onnx"):
    yolox_model = retrieve_yolox_model(current_exp, ckpt_file=ckpt_file)

    torch_inputs = torch.rand((1, 3, 1024, 1024), dtype=torch.float32).to("cuda")

    # Export the model
    torch.onnx.export(
        yolox_model,  # model being run
        torch_inputs,  # model input (or a tuple for multiple inputs)
        onnx_filename,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {
                0: "batch_size",
            },  # variable length axes
            "output": {0: "batch_size"},  # variable length axes
        },
    )


def generate_trt_engine(input_onnx_filename, output_trt_filename, batches=[(32, 32)]):
    """
    This class converts an Onnx model to a TRT model.

    Parameters
    ----------
    input_onnx_filename : `str`
        Path to exported ONNX model.
    output_filename : `str`
        Path to save generated TRT engine.
    batches : `list(tuple)`

    """

    # Local imports to avoid requiring TensorRT to generate the docs
    trt_logger = trt.Logger()

    print(f"Loading ONNX file: {input_onnx_filename}")

    # Otherwise we are creating a new model
    explicit_branch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, builder.create_network(explicit_branch) as network, trt.OnnxParser(
        network, trt_logger
    ) as parser:
        with open(input_onnx_filename, "rb") as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("Count not parse Onnx file. See log.")

        # Now we need to build and serialize the model
        with builder.create_builder_config() as builder_config:
            builder_config.set_flag(trt.BuilderFlag.FP16)

            # Create the optimization files

            for min_batch, max_batch in tqdm(batches, total=len(batches), desc="Generating optimization files"):
                profile = builder.create_optimization_profile()

                # Shape is static for yolox
                min_shape = (min_batch, 3, 1024, 1024)
                shape = (max_batch, 3, 1024, 1024)

                for i in range(network.num_inputs):
                    in_tensor = network.get_input(i)
                    profile.set_shape(in_tensor.name, min=min_shape, opt=shape, max=shape)

                builder_config.add_optimization_profile(profile)

            # Actually build the engine
            print("Building engine. This may take a while...")
            serialized_engine = builder.build_serialized_network(network, builder_config)

            # Now save a copy to prevent building next time
            print(f"Writing engine to: {output_trt_filename}")

            with open(output_trt_filename, "wb") as f:
                f.write(serialized_engine)

            print("Complete!")


def main():
    ckpt_file = "1/best_ckpt.pth"
    onnx_filename = "1/model.onnx"
    trt_filename = "1/model.plan"
    current_exp = yolox_l_9

    # export model to onnx
    export2onnx(current_exp, ckpt_file, onnx_filename=onnx_filename)

    # define batch sizes for trt model
    batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_sizes_all = []
    for batch_size in batch_sizes:
        batch_sizes_all.append((batch_size, batch_size))

    # generate trt engine
    generate_trt_engine(onnx_filename, trt_filename, batches=batch_sizes_all)


if __name__ == "__main__":
    main()
