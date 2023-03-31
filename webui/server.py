from fastapi import FastAPI, Response
from typing import Optional, Tuple, cast
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import io
import logging
# Load embeddings
from yacs.config import CfgNode
from efir.checkpointer import Checkpointer
from efir.utils import CodeBlock, setup_logger, load_config, cfg_node_to_dict
import numpy as np
from efir.registry import Registry
from efir.model.vae import VAE
import pandas as pd
import torch
from dataclasses import dataclass


# Set the CORS policies
origins = [
    "http://172.28.254.48:3000",  # Change this to what you need!
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


setup_logger()
logger = logging.getLogger(__name__)


class Decoder:
    """ little class to load a checkpoint and do stuff
    """

    @dataclass
    class Selection:
        i: int
        j: int
        k: int
        x: float
        y: float
        z: float

    def __init__(
        self,
        run_name = "earthy-dew-137",
        output_dir = "../efir/results",
        config_path = "../efir/configs/mnist_vae_l1.yaml",
        epoch = 21,
        training_labels = "0 2 3 6 7 8 9",
    ) -> None:
        cfg = load_config(config_path)
        device = self.device = cfg.DEVICE

        checkpointer = Checkpointer(output_dir + "/" + run_name)
        results_file = checkpointer.get_inference_results_path(epoch)
        with CodeBlock("Loading model", logger=logger):
            self.model: VAE = cast(
                VAE,
                    checkpointer.load(
                    Registry.build_from_cfg(cfg.MODEL),
                    epoch=epoch,
                ).to(device)
            )
        logger.info(self.model)
        with CodeBlock("Loading embeddings", logger=logger):
            self.embeddings = pd.read_pickle("rsc/embeddings.pkl")

    def get_embedding(self, selection: Selection) -> Tuple[np.ndarray, np.ndarray]:
        mus = {}
        log_vars = {}
        for identifier in ["i", "j", "k"]:
            _df = self.embeddings.iloc[getattr(selection, identifier)]
            mus[identifier] = np.array(_df.mu, dtype=float)
            log_vars[identifier] = np.array(_df.log_var, dtype=float)
        normalization = selection.x + selection.y + selection.z
        mu = (mus["i"] * selection.x) + (mus["j"] * selection.y) + (mus["k"] * selection.z) / normalization
        log_var = (log_vars["i"] * selection.x) + (log_vars["j"] * selection.y) + (log_vars["k"] * selection.z) / normalization
        return (mu, log_var)

    def forward(self, selection: Selection) -> np.ndarray:
        mu, log_var = self.get_embedding(selection=selection)
        mu = torch.from_numpy(mu).float().to(self.device)[None, ...]
        log_var = torch.from_numpy(log_var).float().to(self.device)[None, ...]
        image = self.model.decode(
            self.model.reparameterize(mu, log_var)
        ).cpu().detach().numpy()[0, 0, ...]
        return image

decoder = Decoder()

@app.get("/forward")
def generate_image(i: int, j: int, k: int, x: float, y: float, z: float) -> Response:
    logger.info("generate_image was called")
    img_array = decoder.forward(Decoder.Selection(i=i, j=j, k=k, x=x, y=y, z=z))
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray((255 * (1 - img_array)).astype(np.uint8), mode="L")

    # Convert the PIL Image to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Return the bytes as a response with content type image/png
    return Response(content=img_bytes.getvalue(), media_type="image/png")
