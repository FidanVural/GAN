import torch
import config

input_names = ["actual_input"]
output_names = ["output"]


def onnx_model(gen):
    dummy_input = torch.randn((1, 1, 256, 256)).to(config.DEVICE)

    torch.onnx.export(gen,
                      dummy_input,
                      "gen_model.onnx",
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      )

    print("ONNX model was saved !")

