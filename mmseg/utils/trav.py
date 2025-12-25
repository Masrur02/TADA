import torch

# Traversability priors
CITYSCAPES_TRAV_PRIOR= torch.tensor([
                1.00,  # 0 road
                0.75,  # 1 sidewalk
                0.00,  # 2 building
                0.00,  # 3 wall
                0.00,  # 4 fence
                0.00,  # 5 pole
                0.00,  # 6 traffic light
                0.00,  # 7 traffic sign
                0.25,  # 8 vegetation
                0.50,  # 9 terrain
                0.00,  # 10 sky
                0.00,  # 11 person
                0.00,  # 12 rider
                0.00,  # 13 car
                0.00,  # 14 truck
                0.00,  # 15 bus
                0.00,  # 16 train
                0.00,  # 17 motorcycle
                0.00,  # 18 bicycle
            ], dtype=torch.float32)

FOREST_TRAV_PRIOR = torch.tensor([
    0.65,  # 0 dirt - mostly traversable
    0.45,  # 1 sand - traversable but less stable
    0.80,  # 2 grass - soft but drivable
    0.00,  # 3 tree - obstacle
    0.00,  # 4 pole - obstacle
    0.00,  # 5 water - non-traversable
    0.00,  # 6 sky - irrelevant
    0.00,  # 7 vehicle - obstacle
    0.15,  # 8 object - partially traversable (depends on type)
    1.00,  # 9 asphalt - fully traversable
    0.95,  # 10 gravel - mostly traversable
    0.00,  # 11 building - obstacle
    0.40,  # 12 mulch - soft, somewhat traversable
    0.55,  # 13 rockbed - uneven but partially traversable
    0.10,  # 14 log - obstacle but possibly passable for large robot
    0.00,  # 15 bicycle - obstacle
    0.00,  # 16 person - obstacle
    0.00,  # 17 fence - obstacle
    0.25,  # 18 bush - minor obstacle, sometimes passable
    0.00,  # 19 sign - obstacle
    0.00,  # 20 rock - low traversability, but sometimes drive-over
    0.70,  # 21 bridge - usually traversable (flat surface)
    1.00,  # 22 concrete - good surface
    0.00,  # 23 picnic-table - obstacle
], dtype=torch.float32)

def gt_to_traversability(gt_semantic_seg: torch.Tensor, env: str = "city") -> torch.Tensor:
    
    if env == "city":
        prior = CITYSCAPES_TRAV_PRIOR
    elif env == "forest":
        prior = FOREST_TRAV_PRIOR
    else:
        raise ValueError(f"Unknown environment: {env}")

    device = gt_semantic_seg.device
    prior = prior.to(device)

    gt_ids = gt_semantic_seg.squeeze(1).long()  
    valid_mask = (gt_ids >= 0) & (gt_ids < len(prior))
    trav = torch.zeros_like(gt_ids, dtype=torch.float32, device=device)

    trav[valid_mask] = prior[gt_ids[valid_mask]]

    return trav.unsqueeze(1)  
