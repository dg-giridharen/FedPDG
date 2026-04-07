"""
experiments/comm_cost_analysis.py — Communication Efficiency Analysis
Computes communication overhead for models and aggregates F1 to yield tradeoff ratio.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from models.transformer_encoder import TabularTransformerEncoder

def run_comm_analysis():
    config = Config()
    
    # 1. Theoretical Parameter Count Size
    input_dim = 78 # CICIDS2017
    num_classes = 15
    model = TabularTransformerEncoder(
        input_dim=input_dim, embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS,
    )
    model.build_classifier(num_classes)
    
    model_params = sum(p.numel() for p in model.parameters())
    model_bytes = model_params * 4  # float32
    
    print(f"\n--- Model Size Analysis ---")
    print(f"  Model Parameters: {model_params:,d} parameters")
    print(f"  Base Client Transmit per Round (FLAME/FedAvg/Krum): {model_bytes/1024:.2f} KB")

    # FedPDG transmits Prototypes + Model
    # Prototypes = num_classes * embed_dim 
    proto_params = num_classes * config.EMBED_DIM
    proto_bytes = proto_params * 4
    
    fedpdg_bytes = model_bytes + proto_bytes
    print(f"  FedPDG Prototypes: {proto_params:,d} parameters ({proto_bytes/1024:.2f} KB)")
    print(f"  FedPDG Client Transmit per Round: {fedpdg_bytes/1024:.2f} KB")
    print(f"  FedPDG Communication Overhead: {(proto_bytes/model_bytes)*100:.2f}%")

    # This theoretical analysis can be used in your paper's table!
    print("\n--- Tradeoff Calculation Example ---")
    print("  Assuming FedPDG F1=0.86 and FedAvg F1=0.29 (due to byzantine attack)")
    print(f"  FedPDG Ratio: 0.86 / ({fedpdg_bytes/1024:.2f} KB) = {0.86/(fedpdg_bytes/1024):.6f}")
    print(f"  FedAvg Ratio: 0.29 / ({model_bytes/1024:.2f} KB) = {0.29/(model_bytes/1024):.6f}")

if __name__ == '__main__':
    run_comm_analysis()
