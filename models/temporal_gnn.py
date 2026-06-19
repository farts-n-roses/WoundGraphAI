# =========================================================
# TEMPORAL GRAPH NEURAL NETWORK
# =========================================================

# Learns wound healing progression from spatio-temporal tissue graphs.
# Input: sequence of tissue graphs
# Output: healing prediction score

# =========================================================
# IMPORTS
# =========================================================

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from processing.graph_builder import (run_graph_builder_from_arrays)

from processing.temporal_simulator import (
    generate_healing_sequence,
    binarize_mask
)
from config import DEVICE

# =========================================================
# TEMPORAL GNN MODEL
# =========================================================

class TemporalGNN(torch.nn.Module):

    def __init__(
        self,
        in_channels=5,
        hidden_channels=64,
        out_channels=1
    ):

        super().__init__()

        # =================================================
        # GRAPH CONVOLUTION LAYERS
        # =================================================

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        # =================================================
        # TEMPORAL AGGREGATION
        # =================================================

        self.temporal_fc = Linear(
            hidden_channels,
            hidden_channels
        )

        # =================================================
        # FINAL PREDICTION
        # =================================================

        self.output_fc = Linear(
            hidden_channels,
            out_channels
        )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(self, graph_sequence):

        temporal_embeddings = []

        # =================================================
        # PROCESS EACH GRAPH
        # =================================================

        for graph in graph_sequence:

            x = graph.x

            edge_index = graph.edge_index

            # =============================================
            # GCN LAYER 1
            # =============================================

            x = self.conv1(
                x,
                edge_index
            )

            x = F.relu(x)

            # =============================================
            # GCN LAYER 2
            # =============================================

            x = self.conv2(
                x,
                edge_index
            )

            x = F.relu(x)

            # =============================================
            # GLOBAL REPRESENTATION
            # =============================================

            graph_embedding = torch.mean(
                x,
                dim=0
            )

            temporal_embeddings.append(graph_embedding)

        # =================================================
        # STACK TEMPORAL EMBEDDINGS
        # =================================================

        temporal_tensor = torch.stack(temporal_embeddings)

        # =================================================
        # TEMPORAL AGGREGATION
        # =================================================

        temporal_representation = torch.mean(
            temporal_tensor,
            dim=0
        )

        temporal_representation = self.temporal_fc(temporal_representation)

        temporal_representation = F.relu(temporal_representation)

        # =================================================
        # FINAL PREDICTION
        # =================================================

        output = self.output_fc(temporal_representation)

        output = torch.sigmoid(output)

        return output

# =========================================================
# CREATE TEMPORAL GRAPH SEQUENCE
# =========================================================

# =========================================================
# CREATE TEMPORAL GRAPH SEQUENCE
# =========================================================

def create_graph_sequence():

    graph_sequence = []

    # =====================================================
    # LOAD ORIGINAL IMAGE
    # =====================================================

    image_path = (
        "dataset/images/train_images/fusc_0016.png"
    )

    mask_path = (
        "dataset/masks/train_masks/fusc_0016.png"
    )

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    import cv2

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # LOAD MASK
    # =====================================================

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    mask = binarize_mask(mask)

    # =====================================================
    # GENERATE TEMPORAL MASK SEQUENCE
    # =====================================================

    simulated_masks = generate_healing_sequence(
        mask,
        healing_rate=0.12
    )

    # =====================================================
    # CONVERT EACH MASK TO GRAPH
    # =====================================================

    for simulated_mask in simulated_masks:

        graph_data = (
            run_graph_builder_from_arrays(
                image=image,
                mask=simulated_mask,
                visualize=False,
                verbose=False
            )
        )

        graph_sequence.append(
            graph_data
        )

    return graph_sequence

# =========================================================
# TEST MODEL
# =========================================================

def test_temporal_gnn():

    separator = "=============================="

    print(f"\n{separator}")

    print("CREATING TEMPORAL GRAPH SEQUENCE")

    print(f"{separator}\n")

    graph_sequence = create_graph_sequence()

    print(f"\n{separator}")

    print("INITIALIZING TEMPORAL GNN")

    print(f"{separator}\n")

    model = TemporalGNN().to(DEVICE)

    # =====================================================
    # FORWARD PASS
    # =====================================================

    prediction = model(graph_sequence)

    print(f"\n{separator}")

    print("MODEL OUTPUT")

    print(f"{separator}\n")

    print(
        f"Predicted Healing Score: "
        f"{prediction.item():.4f}"
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    if prediction.item() > 0.8:

        print(
            "\nInterpretation: "
            "Strong healing progression"
        )

    elif prediction.item() > 0.5:

        print(
            "\nInterpretation: "
            "Moderate healing progression"
        )

    else:

        print(
            "\nInterpretation: "
            "Delayed healing risk"
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    test_temporal_gnn()
