# Neural Network from Scratch with NumPy

![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A complete implementation of a feedforward neural network from scratch using only NumPy. This project demonstrates the fundamental mathematics behind neural networks and how they learn from data.

## Features

- Pure NumPy implementation (no ML frameworks)
- Modular architecture with forward/backward propagation
- Sigmoid activation functions
- Mean Squared Error (MSE) loss
- Gradient descent optimization
- Real-time training visualization
- XOR-like pattern learning demonstration

## Mathematical Foundations

The implementation covers all core neural network concepts:

- Forward propagation:
  ```
  h = σ(XW₁ + b₁)
  ŷ = σ(hW₂ + b₂)
  ```
  
- Backpropagation with chain rule:
  ```
  δ₂ = (y - ŷ) ⊙ σ'(ŷ)
  δ₁ = (δ₂W₂ᵀ) ⊙ σ'(h)
  ```
  
- Gradient descent updates:
  ```
  W = W - η(∂L/∂W)
  b = b - η(∂L/∂b)
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nn-from-scratch.git
   cd nn-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

## Usage

Run the neural network training:
```bash
python neural_network.py
```

The script will:
1. Train on an XOR-like dataset
2. Display real-time loss visualization
3. Show test predictions after training

## Sample Output

```
Training the neural network...
Epoch 0, Loss: 0.2500
Epoch 1000, Loss: 0.1002
...
Epoch 19000, Loss: 0.0051

Test predictions:
Input: [0.3 0.7], Predicted: 0.8912
Input: [0.7 0.3], Predicted: 0.8924
```

## Customization

Modify these parameters in `neural_network.py`:
- `hidden_size`: Number of neurons in hidden layer
- `epochs`: Training iterations
- `learning_rate`: Gradient descent step size
- Training data in `X` and `y` arrays

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by fundamental deep learning principles
- NumPy for efficient matrix operations
- Matplotlib for visualization

This README:
1. Uses standard GitHub formatting
2. Includes badges for key technologies
3. Clearly explains what the project does
4. Shows installation/usage instructions
5. Includes mathematical notation
6. Provides customization guidance
7. Has proper licensing information
