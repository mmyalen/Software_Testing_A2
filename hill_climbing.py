"""
Assignment 2 – Adversarial Image Attack via Hill Climbing

You MUST implement:
    - compute_fitness
    - mutate_seed
    - select_best
    - hill_climb

DO NOT change function signatures.
"""

# Uncomment for old MacBook
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
# import tensorflow as tf
# # Disable GPU
# tf.config.set_visible_devices([], 'GPU')

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import array_to_img, load_img, img_to_array
import time 


# ============================================================
# 1. FITNESS FUNCTION
# ============================================================

def compute_fitness(
    image_array: np.ndarray,
    model,
    target_label: str
) -> float:
    """
    Compute fitness of an image for hill climbing.

    Fitness definition (LOWER is better):
        - If the model predicts target_label:
              fitness = probability(target_label)
        - Otherwise:
              fitness = -probability(predicted_label)
    """
    # predict class probabilities
    img_batch = np.expand_dims(image_array, axis=0)
    predictions = model.predict(img_batch)
    decoded_pred = decode_predictions(predictions, top=5)[0]

    # values from class with the highest probability
    predicted_label = decoded_pred[0][1]
    predicted_prob = decoded_pred[0][2]

    # fitness 
    target_prob = None
    for _, label, prob in decoded_pred:
        if label == target_label:
            target_prob = prob
            break

    if predicted_label == target_label:
        return target_prob 
    else:
        return -predicted_prob


# ============================================================
# 2. MUTATION FUNCTION
# ============================================================

def mutate_seed(
    seed: np.ndarray,
    epsilon: float
) -> List[np.ndarray]:
    """
    Produce ANY NUMBER of mutated neighbors.

    Students may implement ANY mutation strategy:
        - modify 1 pixel
        - modify multiple pixels
        - patch-based mutation
        - channel-based mutation
        - gaussian noise (clipped)
        - etc.

    BUT EVERY neighbor must satisfy the L∞ constraint:

        For all pixels i,j,c:
            |neighbor[i,j,c] - seed[i,j,c]| <= 255 * epsilon

    Requirements:
        ✓ Return a list of neighbors: [neighbor1, neighbor2, ..., neighborK]
        ✓ K can be ANY size ≥ 1
        ✓ Neighbors must be deep copies of seed
        ✓ Pixel values must remain in [0, 255]
        ✓ Must obey the L∞ bound exactly

    Args:
        seed (np.ndarray): input image
        epsilon (float): allowed perturbation budget

    Returns:
        List[np.ndarray]: mutated neighbors
    """
    
    neighbors = []
    num_neighbors = 5 
    max_perturb = 255 * epsilon

    for _ in range(num_neighbors):
        noise = np.random.uniform(-max_perturb, max_perturb, seed.shape)
        neighbor = np.clip(seed + noise, 0, 255)
        
        # L∞ constraint 
        diff = np.clip(neighbor - seed, -max_perturb, max_perturb)

        neighbor = np.clip(seed + diff, 0, 255)
        neighbors.append(neighbor.astype(np.float32).copy())

    return neighbors



# ============================================================
# 3. SELECT BEST CANDIDATE
# ============================================================

def select_best(
    candidates: List[np.ndarray],
    model,
    target_label: str
) -> Tuple[np.ndarray, float]:
    """
    Evaluate fitness for all candidates and return the one with
    the LOWEST fitness score.

    Args:
        candidates (List[np.ndarray])
        model: classifier
        target_label (str)

    Returns:
        (best_image, best_fitness)
    """

    best_fitness = None
    best_image = None

    for cand in candidates:
        fitness = compute_fitness(cand, model, target_label)
        if (best_fitness is None) or (fitness < best_fitness):
            best_fitness = fitness
            best_image = cand

    return best_image, best_fitness


# ============================================================
# 4. HILL-CLIMBING ALGORITHM
# ============================================================

def hill_climb(
    initial_seed: np.ndarray,
    model,
    target_label: str,
    epsilon: float = 0.30,
    iterations: int = 300,
    early_stop: int = 20
) -> Tuple[np.ndarray, float]:
    """
    Main hill-climbing loop.

    Requirements:
        ✓ Start from initial_seed
        ✓ EACH iteration:
              - Generate ANY number of neighbors using mutate_seed()
              - Enforce the SAME L∞ bound relative to initial_seed
              - Add current image to candidates (elitism)
              - Use select_best() to pick the winner
        ✓ Accept new candidate only if fitness improves
        ✓ Stop if:
              - target class is broken confidently, OR
              - no improvement for multiple steps (optional)

    Returns:
        (final_image, final_fitness)
    """

    # Start from initial seed
    current_image = initial_seed.copy()
    current_fitness = compute_fitness(current_image, model, target_label)
    steps_no_improve = 0

    for iteration in range(iterations):
        # Generate neighbors using mutate_seed
        neighbors = mutate_seed(current_image, epsilon)
        
        # Enforce L∞ constraint relative to initial_seed
        constrained_neighbors = []
        for neighbor in neighbors:
            diff = neighbor - initial_seed
            diff = np.clip(diff, -255*epsilon, 255*epsilon)
            constrained_neighbor = np.clip(initial_seed + diff, 0, 255)
            constrained_neighbors.append(constrained_neighbor)
        
        # Add current image to candidates (elitism)
        candidates = [current_image] + constrained_neighbors
        
        # Use select_best to pick the winner
        best_image, best_fitness = select_best(candidates, model, target_label)
        
        # Accept only if fitness improves (lower is better)
        if best_fitness < current_fitness:
            current_image = best_image
            current_fitness = best_fitness
            
            # Check if target class is broken confidently
            img_batch = np.expand_dims(current_image, axis=0)
            predictions = model.predict(img_batch)
            decoded_pred = decode_predictions(predictions, top=1)[0]
            predicted_label = decoded_pred[0][1]
            predicted_prob = decoded_pred[0][2]
            
            if predicted_label != target_label and predicted_prob > 0.9:
                print(f"Target broken at iteration {iteration} with confidence {predicted_prob:.4f}")
                break
        else:
            steps_no_improve += 1
        
        # Optional: early stopping
        if steps_no_improve >= early_stop:
            print(f"No improvement for {early_stop} steps, stopping at iteration {iteration}.")
            break
    
    return current_image, current_fitness


# ============================================================
# 5. PROGRAM ENTRY POINT FOR RUNNING A SINGLE ATTACK
# ============================================================

if __name__ == "__main__":
    # Load classifier
    model = vgg16.VGG16(weights="imagenet")

    # Load JSON describing dataset
    with open("data/image_labels.json") as f:
        image_list = json.load(f)

    stats = {} 

    for image_nr in range(0, 10):
        # Pick first entry
        item = image_list[image_nr]
        image_path = "images/" + item["image"]
        target_label = item["label"]

        stats[target_label] = {} 

        print(f"Loaded image: {image_path}")
        print(f"Target label: {target_label}")

        img = load_img(image_path)
        plt.imshow(img)
        plt.title("Original image")
        plt.show()

        img_array = img_to_array(img)
        seed = img_array.copy()

        # Print baseline top-5 predictions
        print("\nBaseline predictions (top-5):")
        preds = model.predict(np.expand_dims(seed, axis=0))
        for cl in decode_predictions(preds, top=5)[0]:
            print(f"{cl[1]:20s}  prob={cl[2]:.5f}")

        st_time = time.time() 
        # Run hill climbing attack
        final_img, final_fitness = hill_climb(
            initial_seed=seed,
            model=model,
            target_label=target_label,
            epsilon=0.30,
            iterations=300
        )
        en_time = time.time() 

        print("Attack took:", en_time - st_time, "seconds")
        print("\nFinal fitness:", final_fitness)

        orig = img_to_array(img)
        mod  = img_to_array(final_img)

        changed_pixels = np.any(orig != mod, axis=-1)
        num_changed_pixels = np.sum(changed_pixels)

        linf_distance = np.max(np.abs(orig - mod))
        total_pixels = orig.shape[0] * orig.shape[1]

        print(f"L_inf distance = {linf_distance}\tChanged_pixels = {100.0 * num_changed_pixels / total_pixels}%")
        stats[target_label]['linf'] = linf_distance 
        stats[target_label]['pixels'] = 100.0 * num_changed_pixels / total_pixels

        # Print final predictions
        final_preds = model.predict(np.expand_dims(final_img, axis=0))
        print("\nFinal predictions:")
        for cl in decode_predictions(final_preds, top=5)[0]:
            print(cl)

        plt.imshow(array_to_img(final_img))
        plt.title(f"{decode_predictions(final_preds, top=5)[0][0][1]} - fitness={final_fitness:.4f} in {int((en_time - st_time) * 100 // 120)} iters")
        plt.savefig(f'hc_results/{item["image"]}_hc.png')
        plt.show()


    for label, ss in stats.items():
        print(label)
        print(ss) 
        print() 
        