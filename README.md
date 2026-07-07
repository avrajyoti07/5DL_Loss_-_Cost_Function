# Loss & Cost Functions — From Scratch (MAE and Log Loss)

🔗 **Repo:** [github.com/avrajyoti07/5DL_Loss_-_Cost_Function](https://github.com/avrajyoti07/5DL_Loss_-_Cost_Function)

A from-scratch look at two core **loss functions** used to train neural networks — **Mean Absolute Error (MAE)** and **Log Loss (Binary Cross-Entropy)** — built using only `numpy`. This is the fifth project in my Deep Learning starter phase, focused on understanding *how a network measures how wrong it is* before that error gets used to update weights during training.

## What's in this repo

| File | Description |
|---|---|
| `4Loss_OR_Cost_Func.ipynb` | Main notebook — implements MAE and Log Loss from scratch, both manually (loop) and vectorized |
| `README.md` | You're here |

## What the notebook does

1. **Sets up sample data** — a `y_predicted` array and a `y_true` array of 5 values to test both loss functions against
2. **Implements MAE manually** — a `mae()` function that loops through each pair, sums the absolute differences, prints the total error, and divides by the count
3. **Vectorizes MAE** — shows the same result using `np.mean(np.abs(y_predicted - y_true))`, matching the loop-based version exactly
4. **Demonstrates a numerical pitfall** — computes `np.log()` on a value extremely close to `0`, showing how the loss can blow up toward `-∞` if a predicted probability ever hits exactly `0` or `1`
5. **Fixes it with epsilon clipping** — clips predictions into a safe range `[1e-15, 1 - 1e-15]` using `max()`/`min()`, so `log()` never sees a literal `0` or `1`
6. **Implements Log Loss (Binary Cross-Entropy) manually** — combines the clipping step with the standard log loss formula, then wraps the whole thing into a single reusable `log_loss()` function

## Results on the sample data

| Loss function | Result |
|---|---|
| MAE | `0.5` (total absolute error: `2.5` across 5 samples) |
| Log Loss (Binary Cross-Entropy) | `17.27` |

> **Note:** the log loss value here is unusually high because in the sample data, `y_predicted` holds hard `0`/`1` values while `y_true` holds fractional values — the reverse of the usual convention (`y_true` = actual labels, `y_predicted` = predicted probabilities). That's fine for exercising the math, but worth keeping straight if you reuse these functions elsewhere — swapping the arguments the "normal" way around will give a very different (and much smaller) loss.

## Concepts covered

- **MAE** — average of the absolute differences between predicted and true values; simple, robust to the sign of the error, easy to interpret
- **Log Loss / Binary Cross-Entropy** — the standard loss for binary classification, which heavily penalizes confident-but-wrong predictions (predicting `~1` when the true label is `0`, or vice versa)
- **Numerical stability** — why `log(0)` is undefined/`-inf` in practice, and why real implementations (including the one inside `keras`/`tensorflow`) clip predictions away from the exact `0`/`1` boundary before taking a log
- Writing the same computation two ways — a manual Python loop vs. a vectorized `numpy` one-liner — and confirming they agree

## Tech stack

- Python 3
- `numpy`

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/avrajyoti07/5DL_Loss_-_Cost_Function.git
cd 5DL_Loss_-_Cost_Function
```

**2. Install dependencies**
```bash
pip install numpy
```

**3. Run the notebook**
```bash
jupyter notebook 4Loss_OR_Cost_Func.ipynb
```
Run all cells top to bottom.

## Key takeaway

A loss function isn't just a formula — it needs to be numerically safe. Log loss looks simple on paper, but a naive implementation crashes or returns `-inf` the moment a prediction is exactly `0` or `1`. Clipping predictions into `[epsilon, 1 - epsilon]` before taking a log is a small detail that's easy to miss but essential for a real, working training loop.

## Next steps

- [ ] Implement Mean Squared Error (MSE) and compare its behavior to MAE on the same data
- [ ] Extend Log Loss to **Categorical Cross-Entropy** for multi-class problems
- [ ] Plug these loss functions into an actual gradient descent update step
- [ ] Compare custom implementations against `keras.losses` built-ins to confirm they match

## Related projects in this series

- [1. Perceptron Demo](https://github.com/avrajyoti07/1DL_Perceptron_demo)
- [2. Hand Digits Classification](https://github.com/avrajyoti07/2DL_Hand_Digits_Classification)
- [3. Activation Functions](https://github.com/avrajyoti07/3DL_Activation_Function)
- [4. Matrix Basics](https://github.com/avrajyoti07/4DL_Matrix_Basics)

## Author

Built by [avrajyoti07](https://github.com/avrajyoti07) as part of a Deep Learning learning journey.
Feel free to fork, star, or open an issue if you spot something to improve!

## License

This project is open source under the [MIT License](LICENSE).
